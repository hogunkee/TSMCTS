import time
import torch
import numpy as np

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WILLOW_BERRY():
    def __init__(self, b_value, b_state, n_child, parent_idx=-1):        
        self.init_value = b_value
        self.berry_state = b_state
        self.n_child = n_child
        self.parent_idx = parent_idx
        
        self.berry_value = b_value
        self.rollout_value = 0.0
        self.child_count, self.child_value, self.child_idx = self.make_child_info()
        
        
    def make_child_info(self):
        return np.zeros(self.n_child), np.zeros(self.n_child), -np.ones(self.n_child)
        
    def get_value(self):
        return self.berry_value

    def get_state(self):
        return np.copy(self.berry_state)

    def update_value(self):
        self.berry_value = max(self.init_value, np.amax(self.child_value), self.rollout_value)

    def update_rollout_value(self, reward):
        self.rollout_value = reward
        
    def get_info(self):
        return np.copy(self.child_count), np.copy(self.child_value), np.copy(self.child_idx)
        
    def get_child_index(self, child_idx):
        return int(self.child_idx[child_idx])
        
    def link_child(self, child_idx, berry_idx):
        self.child_idx[child_idx] = berry_idx
        
    def update_child(self, child_idx, child_value):
        self.child_count[child_idx] += 1
        self.child_value[child_idx] = child_value
        self.update_value()
        
    
              
class WILLOW():
    def __init__(
            self,
            init_state,
            palload_model,
            patch_converter,
            max_depth=10,
            config=None,
            info=None,
        ):
        self.init_state = np.array(init_state)
        self.palload_model = palload_model
        self.patch_converter = patch_converter
        self.n_rotations = config.num_rotations
        self.max_depth, self.searched_depth = max_depth, 0
        self.resolution = np.shape(init_state)[-1]
        self.pallet_size = config.resolution
        self.n_childrun = config.num_rotations*self.resolution*self.resolution
        self.config = config
        self.info = info
        
        self.length_unit = 0.10
        self.box_scale = 10.0
        
        # make root node
        root_berry = WILLOW_BERRY(0.0, init_state, self.n_childrun, fixed_block=True)
        # make node list
        self.berry_list = [root_berry]

    def do_mcts(self, n_tree_iter, n_k_iter, box_list, print_info=True):
        iter_list = []
        search_time_s = time.time()

        n_observed = len(box_list)
        p_projection = 0.0
        
        for n in range(int(n_tree_iter)):
            current_berry, current_idx = self.berry_list[0], 0
            current_state = np.copy(self.init_state)
            current_depth = 0
            cumulative_reward = 0.0
            
            b_history = []
            while True:
                if current_depth >= n_observed:
                    current_box = np.random.choice(
                        [0.2, 0.3, 0.4, 0.5], 3, True
                    )
                    fixed_block = False
                else:
                    current_box = np.copy(box_list[current_depth])
                    fixed_block = True

                box_size = np.ceil(current_box*self.box_scale)/self.box_scale
                cumulative_reward += box_size[0]*box_size[1]*box_size[2]*10.0

                berry_idx, new_one, berry_value = self.choose_child(current_berry, current_state, current_box, cumulative_reward, p_projection=p_projection)
                b_history.append([current_berry, berry_idx])

                # get next state
                if new_one: # make node
                    do_rollout = True
                    next_state = self.update_state(current_state, current_box, berry_idx)
                    next_berry, next_idx = self.add_berry(next_state, current_idx, berry_idx, berry_value, fixed_block)
                else:
                    do_rollout = False
                    next_idx = current_berry.get_child_index(berry_idx)
                    assert next_idx >= 0
                    next_berry = self.berry_list[next_idx]
                    next_state = next_berry.get_state()

                # update information
                current_berry, current_idx, current_state = next_berry, next_idx, next_state
                current_depth += 1

                if berry_value == 0.0:
                    break
                
                if current_depth > self.max_depth:
                    iter_type = "maxstep"
                    break
                
                # MCTS: ROLLOUT
                if do_rollout:
                    roll_value = self.perform_rollout(current_state, current_depth, cumulative_reward, box_list, p_projection)
                    current_berry.update_rollout_value(roll_value)
                    current_berry.update_value()
                    iter_type = "rollout"
                    break
                # MCTS: SIMULATION
                else:
                    pass

            # MCTS: BACKPROPAGATION
            b_history.append([current_berry, -1])
            self.berry_backpropagation(b_history)
            
            root_berry = self.berry_list[0]
            root_value = root_berry.get_value()

            if self.searched_depth < current_depth:
                self.searched_depth = current_depth
            
            num_bars = 40
            progress_ = int((n+1)/n_tree_iter*num_bars)
            percent_ = int((n+1)/n_tree_iter*100)

            search_time_e = time.time()
            self.search_time = search_time_e-search_time_s

            print_line = '  [MCTS][{:02d}S][Progress {}{}:{:3d}%] Root-Berry {:.4f}+{:.4f} | Max-Depth {} | Time {:.1f}s    '\
                .format(self.info["n_steps"],'█'*progress_, ' '*(num_bars-progress_), percent_, self.info["pack_factor"]*10.0, root_value, self.searched_depth, self.search_time)
            print(print_line, end='\r')
        
        if print_info:
            print(print_line)
        else:
            print(" "*120, end='\r')
        
    def optimal_selection(self, mask=None):
        root_berry = self.berry_list[0]
        c_values, r_values, _ = root_berry.get_info()

        if mask is not None:
            mask = np.reshape(mask, (-1))
            r_values *= mask
        best_idx = np.argmax(r_values)

        rot_ = best_idx // (self.resolution*self.resolution)
        y_ = (best_idx % (self.resolution*self.resolution)) // self.resolution
        x_ = (best_idx % (self.resolution*self.resolution)) % self.resolution
        return [rot_, y_, x_]
    
    def convert_state2patch(self, state, block):
        cum_state = generate_cumulative_state(state)
        level_map = np.sum(cum_state, axis=0, keepdims=True)
        level_map = np.expand_dims(level_map, 0)
        level_map = torch.from_numpy(level_map).int()
        level_patches = self.patch_converter.convert2patch(level_map)
        level_patches = level_patches.cpu().detach().numpy()[0]

        block_state = generate_block_state(block, self.config.patch_size,
                                           n_rotations=self.n_rotations)
        return level_patches, block_state

        
    def add_berry(
            self,
            state,
            parent_index,
            child_index,
            init_value,
            fixed_block,
        ):
        # make new berry
        berry = WILLOW_BERRY(init_value, state, self.n_childrun, parent_index, fixed_block=fixed_block)
        # make node list
        berry_index = len(self.berry_list)
        self.berry_list.append(berry)
        # update parent node's information
        parent_berry = self.berry_list[parent_index]
        parent_berry.link_child(child_index, berry_index)
        return berry, berry_index
    
    def estimate_values(self, state, block, reward=0.0, p_projection=0.0):
        # convert state
        p_state, p_block = self.convert_state2patch(state, block)

        if self.pallet_size == self.resolution:
            p_mask = generate_bound_mask(state, block, n_rotations=self.n_rotations)
            p_mask = generate_solid_mask(state, block, p_mask, n_rotations=self.n_rotations)
        else:
            state_pallet = state[:,:self.pallet_size,:self.pallet_size]
            mask_pallet = generate_bound_mask(state_pallet, block, n_rotations=self.n_rotations)
            mask_pallet = generate_solid_mask(state_pallet, block, mask_pallet, n_rotations=self.n_rotations)

            p_mask = np.zeros((self.n_rotations,self.resolution,self.resolution))
            p_mask[:,:self.pallet_size,:self.pallet_size] = mask_pallet

        # choose action
        _, child_values = self.palload_model.get_action(
            p_state,
            p_block,
            p_mask,
            state_raw=state,
            block_raw=block,
            with_q=True,
            deterministic=True,
            p_project=p_projection
        )
        child_values = (child_values+reward) * p_mask
        return np.reshape(child_values, (-1)), np.reshape(p_mask, (-1))

    def choose_child(self, berry, state, block, reward, p_projection=0.0):
        child_counts, child_values, _ = berry.get_info()
        
        child_values, mask = self.estimate_values(state, block, reward, p_projection)

        child_counts = np.array(child_counts)
        child_values = np.array(child_values)

        count_sum = np.sum(child_counts)
        count_sum = max(count_sum, 1) 
        
        # berry_vals = child_values*0.01 + np.sqrt(2.0*np.log(float(count_sum))/(child_counts+1e-3))
        # berry_vals = child_values*3.0 + np.sqrt(2.0*np.log(float(count_sum))/(child_counts+1e-3))
        # berry_vals = child_values*5.0 + np.sqrt(2.0*np.log(float(count_sum))/(child_counts+1e-3))
        berry_vals = child_values*20.0 + np.sqrt(2.0*np.log(float(count_sum))/(child_counts+1e-3))
        berry_vals *= mask

        max_val = np.amax(berry_vals)
        max_is = np.argwhere(np.array(berry_vals)==max_val).reshape(-1)
        assert len(max_is) > 0
        child_idx = np.random.choice(max_is)  

        if mask[child_idx] == 0:
            a = 1

        if child_counts[child_idx] == 0: new_one = True
        else: new_one = False
        return child_idx, new_one, child_values[child_idx]

    def update_state(self, current_state, current_box, berry_index):
        rot_ = berry_index // (self.resolution*self.resolution)
        y_ = (berry_index % (self.resolution*self.resolution)) // self.resolution
        x_ = (berry_index % (self.resolution*self.resolution)) % self.resolution

        current_box = np.ceil(current_box*self.box_scale)
        size_, _ = block_rotation(current_box, rot_)
        by, bx, bz = size_

        next_block_bound = get_block_bound(y_, x_, by, bx)
        min_y, max_y, min_x, max_x = next_block_bound

        box_height = int(bz/(self.length_unit*self.box_scale))
        assert box_height > 0

        level_map = generate_cumulative_state(current_state)
        level_map = np.sum(level_map, axis=0)

        pre_level = np.max(level_map[max(min_y,0):max_y,max(min_x,0):max_x])

        max_level = np.shape(current_state)[0]

        next_state = np.copy(current_state)
        for h_ in range(box_height):
            level_ = int(pre_level + h_)
            if level_ < max_level:
                next_state[level_,max(min_y,0):max_y,max(min_x,0):max_x] = 1.0
        return next_state

    def perform_rollout(self, current_state, current_depth, current_reward, box_list, p_projection=0.0):
        rollut_value = 0.0
        while current_depth < self.max_depth:
            if current_depth >= len(box_list):
                current_box = np.random.choice(
                    [0.2, 0.3, 0.4, 0.5], 3, True
                )
            else:
                current_box = box_list[current_depth]
            # current_reward += current_box[0]*current_box[1]*current_box[2]*10.0

            child_values, mask = self.estimate_values(current_state, current_box, 0.0, p_projection)
            if np.sum(mask) == 0.0:
                rollut_value = current_reward
                break
            
            current_reward += current_box[0]*current_box[1]*current_box[2]*10.0

            child_idx = np.argmax(child_values)
            rollut_value = np.max(child_values) + current_reward
            current_state = self.update_state(current_state, current_box, child_idx)
            current_depth += 1

        return rollut_value
        #return 0.0
        
    def berry_backpropagation(self, berry_list):
        prev_berry, _ = berry_list[-1]
        for i, (berry, c_idx) in enumerate(reversed(berry_list[:-1])):
            c_value = prev_berry.get_value()
            berry.update_child(c_idx, c_value)
            prev_berry = berry
        
    def get_search_time(self):
        return self.search_time
        
    def print_mcts_result(self):
        root_berry = self.berry_list[0]
        c_values, r_values, _ = root_berry.get_info()
        child_idx = np.argmax(r_values)
        if child_idx == 1:
            print("  [MCTS-Finished] Willow chooses {}-st Berry. :D".format(child_idx))
        elif child_idx == 2:
            print("  [MCTS-Finished] Willow chooses {}-nd Berry. :D".format(child_idx))
        elif child_idx == 3:
            print("  [MCTS-Finished] Willow chooses {}-rd Berry. :D".format(child_idx))
        else:
            print("  [MCTS-Finished] Willow chooses {}-th Berry. :D".format(child_idx))
        #print("                  R-Values: ", r_values)
        #print("                  N-Counts: ", c_values)
        print("                  BEST: {:.4f} / {} ".format(r_values[child_idx],int(c_values[child_idx])))
        print("                  Planning-Time: {:.3f}s".format(self.search_time))
        print()
    
    
    
