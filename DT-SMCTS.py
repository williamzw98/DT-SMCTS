import pandas as pd
import numpy as np
import math
import copy
import csv
import time
import sys
from multiprocessing import Pool
from sorter import Sorter

str_province = \
    {'河北省': ['内蒙古自治区', '山西省', '山东省', '北京市', '天津市', '辽宁省', '河南省'],
     '山东省': ['河北省', '天津市', '山西省', '湖南省', '江苏省'],
     '安徽省': ['山东省', '湖南省', '湖北省', '江西省', '浙江省', '江苏省'],
     '江苏省': ['山东省', '安徽省', '上海市', '浙江省'],
     '上海市': ['江苏省', '浙江省'],
     '浙江省': ['上海市', '安徽省', '江西省', '福建省', '江苏省'],
     '江西省': ['安徽省', '湖北省', '湖南省', '广东省', '福建省', '浙江省'],
     '福建省': ['浙江省', '江西省', '广东省'],
     '黑龙江省': ['内蒙古自治区', '吉林省'],
     '吉林省': ['内蒙古自治区', '黑龙江省'],
     '辽宁省': ['内蒙古自治区', '吉林省', '河北省'],
     '北京市': ['河北省', '天津市'],
     '天津市': ['河北省', '北京市'],
     '海南省': [],
     '台湾省': [],
     '澳门特别行政区': [],
     '香港特别行政区': [],
     '新疆维吾尔自治区': ['西藏自治区', '青海省', '甘肃省'],
     '西藏自治区': ['新疆维吾尔自治区', '青海省', '四川省', '云南省'],
     '青海省': ['西藏自治区', '新疆维吾尔自治区', '甘肃省', '四川省'],
     '甘肃省': ['新疆维吾尔自治区', '青海省', '四川省', '内蒙古自治区', '陕西省', '宁夏回族自治区'],
     '四川省': ['西藏自治区', '青海省', '甘肃省', '陕西省', '重庆市', '贵州省', '云南省'],
     '云南省': ['四川省', '贵州省', '广西壮族自治区'],
     '内蒙古自治区': ['甘肃省', '宁夏回族自治区', '陕西省', '山西省', '河北省', '吉林省', '辽宁省', '黑龙江省'],
     '宁夏回族自治区': ['陕西省', '内蒙古自治区', '甘肃省'],
     '陕西省': ['宁夏回族自治区', '甘肃省', '四川省', '内蒙古自治区', '重庆市', '湖北省', '湖南省', '山西省'],
     '重庆市': ['陕西省', '四川省', '贵州省', '湖南省', '湖北省'],
     '贵州省': ['重庆市', '四川省', '云南省', '广西壮族自治区', '湖南省'],
     '广西壮族自治区': ['云南省', '贵州省', '湖南省', '广东省'],
     '山西省': ['内蒙古自治区', '陕西省', '河南省', '山东省', '河北省'],
     '河南省': ['山西省', '陕西省', '湖北省', '安徽省', '山东省', '河北省'],
     '湖北省': ['河南省', '重庆市', '陕西省', '湖南省', '江西省', '安徽省'],
     '湖南省': ['湖北省', '重庆市', '贵州省', '广西壮族自治区', '广东省', '江西省', '安徽省'],
     '广东省': ['湖南省', '江西省', '福建省', '广西壮族自治区']}

region_str = \
    {'河北省': ['311', '310', '319', '312', '313', '314', '315', '335', '317', '316', '318'],
     '山东省': ['530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '543', '546', '631', '632', '633', '634', '635'],
     '安徽省': ['551', '554', '553', '552', '555', '556', '559', '558', '550', '557', '561', '563', '562', '564', '566'],
     '江苏省': ['025', '516', '518', '517', '527', '515', '514', '513', '511', '519', '510', '512', '523'],
     '上海市': ['021'],
     '浙江省': ['571', '574', '573', '572', '575', '579', '570', '580', '577', '576', '578'],
     '江西省': ['791', '798', '790', '792', '793', '795', '794', '796', '797', '701', '799'],
     '福建省': ['591', '592', '598', '594', '595', '596', '599', '593', '597'],
     '黑龙江省': ['451', '452', '453', '454', '459', '458', '464', '467', '468', '469', '455', '457', '456'],
     '吉林省': ['431', '432', '433', '434', '435', '436', '437', '438', '439'],
     '辽宁省': ['024', '411', '412',  '415', '416', '417', '418', '419', '421', '427', '429'],
     '北京市': ['010'],
     '天津市': ['022'],
     '海南省': ['898', '899', '890'],
     '台湾省': ['0886'],
     '澳门特别行政区': ['0853'],
     '香港特别行政区': ['0852'],
     '新疆维吾尔自治区': ['901', '902', '903', '904', '905', '906', '907', '908', '909', '990', '991', '992', '993', '994', '995', '996', '997', '998', '999'],
     '西藏自治区': ['891', '893', '895', '897', '892', '894', '896'],
     '青海省': ['971', '972', '973', '974', '975', '976', '977', '979', '978', '970'],
     '甘肃省': ['931', '930', '937', '943', '938', '935', '936', '933', '934', '932', '939', '941'],
     '四川省': ['028', '812', '817', '818', '825', '826', '827', '838', '816', '813', '832', '833', '830', '831', '837', '838', '836', '834', '835', '839'],
     '云南省': ['871', '870', '872', '873', '874', '875', '876', '877', '878', '879', '883', '886', '887', '888', '691', '692'],
     '内蒙古自治区': ['471', '472', '476', '479', '470', '475', '477', '482', '17020', '473', '474', '478', '483'],
     '宁夏回族自治区': ['951', '952', '953', '954', '955'],
     '陕西省': ['029', '911', '912', '913', '914', '915', '916', '917', '919'],
     '重庆市': ['023'],
     '贵州省': ['851', '852', '853', '854', '855', '857', '859', '856', '858'],
     '广西壮族自治区': ['770', '771', '772', '773', '774', '775', '776', '778', '779', '777'],
     '山西省': ['351', '352', '353', '354', '355', '349', '356', '358', '357', '359', '350'],
     '河南省': ['370', '371', '372', '374', '375', '376', '378', '379', '373', '391', '392', '393', '377', '394', '395', '396', '398'],
     '湖北省': ['027', '714', '710', '711', '719', '717', '715', '712', '713', '718', '716', '722', '724', '728'],
     '湖南省': ['731', '733', '732', '734', '730', '736', '735', '737', '738', '739', '746', '745', '744', '743'],
     '广东省': ['020', '660', '662', '663', '668', '755', '750', '756', '754', '751', '752', '753', '758', '762', '763', '766', '768', '769', '760', '757', '759']}

class Vertex:
    def __init__(self, key):
        self.id = key
        self.area_set = set()
        self.inset_packagename = set()
        self.connectedTo = set()

    def add_area_list(self):
        for i in region_str[self.id]:
            self.area_set.add(i)

    def add_packagename_node(self, packagename):
        self.inset_packagename.add(packagename)

    def add_neighbor(self):
        for i in str_province[self.id]:
            self.connectedTo.add(i)

    def get_connections(self):
        return self.connectedTo

    def get_id(self):
        return self.id

    def is_package_belongs(self, package_id):
        return package_id in self.area_set

class Graph:
    def __init__(self, region_str):
        self.verList = {}
        self.numVertices = 0
        self.region_str = region_str

    def add_vertex(self, key):
        self.numVertices += 1
        new_vertex = Vertex(key)
        new_vertex.add_neighbor()
        new_vertex.add_area_list()
        self.verList[key] = new_vertex
        return new_vertex

    def get_vertex(self, node):
        if node in self.verList:
            return self.verList[node]
        else:
            return None

    def get_vertices(self):
        return self.verList.keys()

    def __iter__(self):
        return iter(self.verList.values())

    def add_package_region(self, stream_id, package_name):
        flow_id = stream_id[:4]
        for key in self.region_str:
            if flow_id in self.region_str[key]:
                vertex = self.get_vertex(key)
                if vertex:
                    vertex.add_packagename_node(package_name)
                    return True
                else:
                    return False
        return False

    def get_package_region(self, package_name):
        for item in self.verList:
            if package_name in self.verList[item].inset_packagename:
                return item
        return None

    def get_neighboring_packages(self, state, current_cluster, selected_clusters, suffix, mode=0):
        state.current_cluster = current_cluster
        if len(set(selected_clusters)) == n_clusters:
            print("End!")
            return [], []
        package_name = current_cluster
        part = current_cluster.split('-')
        if len(part) >= 2:
            current_flow_id = part[1][:3]
        else:
            current_flow_id = '000'

        for province in str_province.keys():
            self.add_vertex(province)

        for index, row in df.iterrows():
            package = row[0]
            part = package.split('-')
            if len(part)>=2:
                stream_info = part[1]
            else:
                stream_info = '000'
            # stream_info = package.split('-')[1]
            flow_id = stream_info[:3]
            self.add_package_region(flow_id, package)

        current_province = self.get_package_region(package_name)
        available_province_packages = []
        available_neighboring_packages = []

        package_suffix = 'A' if (suffix in range(1, 74) or suffix in range(137, 208)) else 'B'

        if current_province:
            available_province_packages = [pkg for pkg in self.verList[current_province].inset_packagename if pkg not in selected_clusters and pkg.endswith('#' + package_suffix)]
            for neighbor_province in self.verList[current_province].get_connections():
                available_neighboring_packages.extend([pkg for pkg in self.verList[neighbor_province].inset_packagename if pkg not in selected_clusters and pkg.endswith('#' + package_suffix)])

        if not current_province or (not available_province_packages and not available_neighboring_packages):
            all_available_packages = [pkg for pkg in package_info_dict if pkg not in selected_clusters and pkg.endswith('#' + package_suffix)]
            target_proportion = package_info_dict.get(current_cluster, {}).get('proportion', 0)
            closest_package = min(all_available_packages, key=lambda pkg: abs(target_proportion - package_info_dict[pkg]['proportion'])) if all_available_packages else None
            selected_packages = [closest_package] if closest_package else []
        else:
            same_province_packages = sorted(
                available_province_packages,
                key=lambda x: flow_id_similarity(current_flow_id, x.split('-')[1][:3])
            )

            neighboring_packages = sorted(
                available_neighboring_packages,
                key=lambda x: flow_id_similarity(current_flow_id, x.split('-')[1][:3])
            )

            selected_packages = []
            if mode == 0:
                total_packages = 8
                while len(selected_packages) < total_packages:
                    if same_province_packages and len(selected_packages) < 7:
                        selected_packages.append(same_province_packages.pop(0))
                    elif neighboring_packages:
                        selected_packages.append(neighboring_packages.pop(0))
                    else:
                        break
            elif mode == 1:
                if same_province_packages:
                    selected_packages.append(same_province_packages.pop(0))
                elif neighboring_packages:
                    selected_packages.append(neighboring_packages.pop(0))

        action_probs = [(package, package_info_dict[package]['proportion']) for package in selected_packages if package]
        value = [package_info_dict[package]['proportion'] for package in selected_packages if package]

        return action_probs, value


def flow_id_similarity(flow_id1, flow_id2):
    return abs(int(flow_id1[:3]) - int(flow_id2[:3]))

df = pd.read_excel('overall.xlsx')

package_info_dict = {}
percentages = []
package_city_manager = Graph(region_str)


for province in str_province.keys():
    package_city_manager.add_vertex(province)

package_index = 1

for index, row in df.iterrows():

    package_name_total = row[0]  
    proportion = row[1]  

    parts = package_name_total.split('-')  
    if len(parts) >= 2:  
        time_info = parts[0]  
        stream_info = parts[1]  
    else:  
        time_info = 'T4'  
        stream_info = '000'

    flow_id = ''  
    oop = 0
    for i in stream_info:  
        if str.isdigit(i) and oop < 4:  
            flow_id += i  
            oop += 1
        else:
            flow_id = '000'  
            
        package_info = {
            'package_number': package_index,
            'time_sort': time_info,
            'flow_id': flow_id,
            'proportion': proportion,
        }
        package_name = package_name_total 


        percentages.append(proportion)
        package_city_manager.add_package_region(flow_id, package_name)

        package_info_dict[package_name] = package_info

        package_index += 1

n_clusters = len(package_info_dict)

fx = pd.read_excel('fixed.xlsx')

clusters = []
fixed_slots_number = []
fixed_packages = []
fixed_packages = fx.iloc[:, 0].tolist()  
fixed_slots_number = fx.iloc[:, 2].tolist()  
fixed_slots_number_adjusted = [num - 1 for num in fixed_slots_number]
fixed_slots = dict(zip(fixed_slots_number_adjusted, fixed_packages))
selected_clusters = []
for key, value in fixed_slots.items():
    selected_clusters.append(value)
selected_nodes = []
check_clusters = []
check_nodes = []
rollout_results = []
leaf_value_list = []
best_value = -99999999
best_arrangement = []
n_clusters = n_clusters - len(selected_nodes)

class TreeNode(object):

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  
        self._path = []
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._cluster = 'T6-730VA混#A'


    def expand(self, action_priors):                    

        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
                self._children[action]._cluster = action

    def select(self, c_puct):

        for action, node in self._children.items():
            if node._n_visits > 100:
                return action, node
       
            else:
                action, next_node = max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

        return action, next_node

    def update(self, leaf_value):

        self._n_visits += 1

        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):

        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        if self._n_visits == 0:
            self._u = 50000
        else:
            self._u = (c_puct  *
                    np.sqrt((2 * math.log(self._parent._n_visits)) / self._n_visits))
        return self._Q + self._u

    def is_leaf(self):

        return self._children == {}

    def is_root(self):
        return self._parent is None
    
    def get_path(self):
        if self._parent is None:
            return [self._cluster]  

        parent_path = self._parent.get_path()  
        return parent_path + [self._cluster]  
    
    def check_path(self):
        if self._parent is None:
            return [self]
        check_path = self._parent.check_path()
        return check_path + [self]


class Allocation(object):
    def __init__(self, clusters, n_clusters, current_cluster='T6-730VA混#A', c_puct= 0.3):
        self.clusters = clusters
        self.n_clusters = n_clusters
        self._c_puct = c_puct
        self.grid = [None] * n_clusters  
        self.current_index = 19  
        self.current_cluster = current_cluster  
        self.root_node = TreeNode(None, 1.0)
        self.selected_clusters = [None] * n_clusters

    def game_end(self):
        if None in self.grid:
            return False, None
        else:
            return True, self.grid
   
    def do_move(self, grid, current_index, current_cluster, action, mode = 0):
        if self.game_end()[0]:
            return
            
        if mode == 0:
            if current_index < len(grid):
                grid[current_index] = current_cluster
                current_index += 1
                current_cluster = action


        elif mode == 1:
            if current_index < len(grid) and action not in fixed_slots.values():
                grid[current_index] = action
                current_cluster = action
        return current_index, current_cluster, grid

class MCTS(object):

    def __init__(self, c_puct= 0.3, n_playout=8000, current_index = 19, current_cluster = 'T6-730VA混#A'):

        self._root = TreeNode(None, 1.0)
        self._graph_instance = Graph(region_str)
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._current_index = current_index
        self._current_cluster = current_cluster

    def _policy(self, state, current_cluster, selected_clusters, current_index, mode=0):
        
        part = current_cluster.split('-')
        if len(part) >= 2:
            flow_id = part[1][:3]
        else:
            flow_id = '000'

        self._graph_instance.add_package_region(flow_id, package_name)
        return self._graph_instance.get_neighboring_packages(state, current_cluster, selected_clusters,current_index, mode)      
        
    def game_end(self, grid):
        if None in grid:
            return False, None
        else:
            return True, grid

    def update_with_move(self, last_move):

        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
    
    def digital_twin(self, standard_rollout_input):
        sorter_instance = Sorter()
        avg_capacity, peak_capacity = sorter_instance.update_mapping_and_run(standard_rollout_input)
        return avg_capacity, peak_capacity
    
    def _rollout(self, state, current_cluster, grid):
        
        for i in range(len(state.grid)):
            if state.grid[i] is None:
                current_index = i
                break
        state.current_index = current_index
        state.current_cluster = current_cluster
        state.selected_clusters = selected_clusters

        
        while(1):
            end, rollout_grid = self.game_end(state.grid)
            if end:
                break
            action_probs, _ = self._policy(state, state.current_cluster,state.selected_clusters,state.current_index, mode = 1)

            relay = action_probs[0]
            max_action = relay[0]
            state.selected_clusters[:len(selected_clusters)] = selected_clusters
            state.current_index, state.current_cluster, rollout_grid = state.do_move(state.grid, state.current_index, state.current_cluster,max_action, mode = 1)
            state.selected_clusters.append(state.current_cluster)
            for i in range(len(rollout_grid)):
                if rollout_grid[i] is None:
                    state.current_index = i
                    state.current_cluster = grid[i-1]
                    break                    

        return rollout_grid

    def __str__(self):
        return "MCTS"
  

def main():

    state = Allocation(clusters, n_clusters)  

    mcts = MCTS(c_puct= 0.3, n_playout=8000,current_cluster='T6-730VA混#A', current_index=19)

    for n in range(mcts._n_playout):
        state_copy = copy.deepcopy(state)
        number = -99
        node = mcts._root
        playout_grid = [None] * n_clusters
        for slot, package in fixed_slots.items():
            playout_grid[slot] = package
        global selected_clusters   
        path = []
        while(1):
            if node.is_leaf():
                break

            action, node = node.select(mcts._c_puct)
       
            mcts._current_cluster = action

        path = node.get_path()

        for p in path:
            index = 0
            while index < len(playout_grid):
                if playout_grid[index] is None:
                    playout_grid[index] = p
                    break
                index += 1

        for key, value in fixed_slots.items():
            path.append(value)
        selected_clusters = path

        prefixed_cluster = None
        for i in range(len(playout_grid)):
            if playout_grid[i] is None:
                if i > 0:
                    prefixed_cluster = playout_grid[i-1]  
                    suffix = i
                break
        if prefixed_cluster is not None:
            action_probs, _ = mcts._policy(state_copy, prefixed_cluster, selected_clusters, suffix, mode = 0)    

        end, final_grid = mcts.game_end(playout_grid)      

        if not end:
            node.expand(action_probs)
 
            state_copy.current_cluster = prefixed_cluster
            state_copy.grid = playout_grid  

            state_copy.grid = mcts._rollout(state_copy, state_copy.current_cluster, state_copy.grid)
            rollout_dict = {index + 1: value for index, value in enumerate(state_copy.grid)}

            standard_rollout_input = {}
            for key, value in rollout_dict.items():
                value = value.split('#')[0]  
                parts = value.split('*')[0]                

                multiple_values = set(parts.split())
                if 'E-mpty' not in multiple_values:
                    standard_rollout_input[key] = multiple_values           
            avg_capacity, peak_capacity = mcts.digital_twin(standard_rollout_input) 
            leaf_value = 0.7*avg_capacity + 0.3*peak_capacity

            global rollout_results
            rollout_results.append(leaf_value)
            node.update_recursive(leaf_value)
            leaf_value_list.append(leaf_value)
            global best_value
            global best_arrangement
            if leaf_value > best_value:
                best_value = leaf_value
                best_arrangement = state_copy.grid
                best_arrangement_dict = {index + 1: value for index, value in enumerate(best_arrangement)}
                print("Best arrangement:",best_arrangement_dict)
                print("Best value:",best_value)
        else:        
            final_grid_dict = {index + 1: value for index, value in enumerate(final_grid)}
            final_rollout_input = {}
            for key, value in final_grid_dict.items():
                value = value.split('#')[0]  
                parts = value.split('*')[0]                  

                multiple_values = set(parts.split())
                if 'E-mpty' not in multiple_values:
                    final_rollout_input[key] = multiple_values 
            avg_capacity, peak_capacity = mcts.digital_twin(final_rollout_input)  
            leaf_value = 0.7*avg_capacity + 0.3*peak_capacity
            rollout_results.append(leaf_value)
            leaf_value_list.append(leaf_value)
            number = 100        
            
            selected_clusters = []
            if number == 100:
                break

if __name__ == '__main__':
    main()
