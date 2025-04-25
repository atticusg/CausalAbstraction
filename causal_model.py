from collections import defaultdict
import random
import copy
import itertools
from ipycytoscape import CytoscapeWidget
from datasets import Dataset, load_dataset
import networkx as nx
import matplotlib.pyplot as plt


class CausalModel:
    """
    A class to represent a causal model with variables, values, parents, and mechanisms.
    Attributes:
    -----------
    variables : list
        A list of variables in the causal model.
    values : dict
        A dictionary mapping each variable to its possible values.
    parents : dict
        A dictionary mapping each variable to its parent variables.
    mechanisms : dict
        A dictionary mapping each variable to its causal mechanism.
    input_loader : function, optional
        A function to load inputs (default is None).
    input_dumper : function, optional
        A function to dump inputs (default is None).
    output_dumper : function, optional
        A function to dump outputs (default is None).
    print_pos : dict, optional
        A dictionary specifying positions for plotting (default is None).
    """
    def __init__(
        self,
        variables,
        values,
        parents,
        mechanisms,
        print_pos=None,
    ):
        self.variables = variables
        self.values = values
        self.parents = parents
        self.mechanisms = mechanisms


        # create children and verify that 
        self.children = {var: [] for var in variables}
        for variable in variables:
            assert variable in self.parents
            for parent in self.parents[variable]:
                self.children[parent].append(variable)

        # find inputs and outputs
        self.inputs = [var for var in self.variables if len(parents[var]) == 0]
        self.outputs = copy.deepcopy(variables)
        for child in variables:
            for parent in parents[child]:
                if parent in self.outputs:
                    self.outputs.remove(parent)

        # generate timesteps
        self.timesteps = {input: 0 for input in self.inputs}
        step = 1
        change = True
        while change:
            change = False
            copytimesteps = copy.deepcopy(self.timesteps)
            for parent in self.timesteps:
                if self.timesteps[parent] == step - 1:
                    for child in self.children[parent]:
                        copytimesteps[child] = step
                        change = True
            self.timesteps = copytimesteps
            step += 1
        self.end_time = step - 2
        for output in self.outputs:
            self.timesteps[output] = self.end_time

        # verify that the model is valid
        for variable in self.variables:
            try:
                assert variable in self.values
            except:
                raise ValueError(f"Variable {variable} not in values")
            try:
                assert variable in self.children
            except:
                raise ValueError(f"Variable {variable} not in children")
            try:
                assert variable in self.mechanisms
            except:
                raise ValueError(f"Variable {variable} not in mechanisms")
            try:
                assert variable in self.timesteps
            except:
                raise ValueError(f"Variable {variable} not in timesteps")

            for variable2 in copy.copy(self.variables):
                if variable2 in self.parents[variable]:
                    try:
                        assert variable in self.children[variable2]
                    except:
                        raise ValueError(f"Variable {variable} not in children of {variable2}")
                    try:
                        assert self.timesteps[variable2] < self.timesteps[variable]
                    except:
                        raise ValueError(f"Variable {variable2} has a later timestep than {variable}")
                if variable2 in self.children[variable]:
                    try:
                        assert variable in parents[variable2]
                    except:
                        raise ValueError(f"Variable {variable} not in parents of {variable2}")
                    try:
                        assert self.timesteps[variable2] > self.timesteps[variable]
                    except:
                        raise ValueError(f"Variable {variable2} has an earlier timestep than {variable}")
        
        # sort variables by timestep
        self.variables.sort(key=lambda x: self.timesteps[x])


        # set positions for plotting
        self.print_pos = print_pos
        width = {_: 0 for _ in range(len(self.variables))}
        if self.print_pos == None:
            self.print_pos = dict()
        for var in self.variables:
            if var not in self.print_pos:
                self.print_pos[var] = (width[self.timesteps[var]], self.timesteps[var])
                width[self.timesteps[var]] += 1

        # Initializing the equivalence classes of children values
        # that produce a given parent value is expensive
        self.equiv_classes = {}

    # FUNCTIONS FOR RUNNING THE MODEL

    def run_forward(self, intervention=None):
        total_setting = defaultdict(None)
        length = len(list(total_setting.keys()))
        while length != len(self.variables):
            for variable in self.variables:
                for variable2 in self.parents[variable]:
                    if variable2 not in total_setting:
                        continue
                if intervention is not None and variable in intervention:
                    total_setting[variable] = intervention[variable]
                else:
                    total_setting[variable] = self.mechanisms[variable](
                        *[total_setting[parent] for parent in self.parents[variable]]
                    )
            length = len(list(total_setting.keys()))
        return total_setting

    def run_interchange(self, input, counterfactual_inputs):
        interchange_intervention = copy.deepcopy(input)
        for var in counterfactual_inputs:
            setting = self.run_forward(counterfactual_inputs[var])
            interchange_intervention[var] = setting[var]
        return self.run_forward(interchange_intervention)


# FUNCTIONS FOR SAMPLING INPUTS AND GENERATING DATASETS

    def sample_intervention(self, filter=None):
        filter = filter if filter is not None else lambda x: True
        intervention = {}
        while not filter(intervention):
            intervention = {}
            while len(intervention.keys()) == 0:
                for var in self.variables:
                    if var in self.inputs or var in self.outputs:
                        continue
                    if random.choice([0, 1]) == 0:
                        intervention[var] = random.choice(self.values[var])
        return intervention

    def sample_input(self, filter=None):
        filter = filter if filter is not None else lambda x: True
        input = {var: random.sample(self.values[var], 1)[0] for var in self.inputs}
        total = self.run_forward(intervention=input)
        while not filter(total):
            input = {var: random.sample(self.values[var], 1)[0] for var in self.inputs}
            total = self.run_forward(intervention=input)
        return input

    def generate_factual_dataset(self, size, input_sampler=None, filter=None):
        if input_sampler is None:
            input_sampler = CausalModel.sample_input
        inputs = []
        while len(inputs) < size:
            inp = input_sampler()
            if filter is None or filter(inp):
                inputs.append(self.input_dumper(inp))
        # Create and return a Hugging Face Dataset with a single "input" field.
        return Dataset.from_dict({"input": inputs})

    def generate_counterfactual_dataset(self, size, counterfactual_sampler, filter=None):
        inputs = []
        counterfactuals = []
        while len(inputs) < size:
            sample = counterfactual_sampler()  # sample is a dict with keys "input" and "counterfactual_inputs"
            if filter is None or filter(sample):
                inputs.append(sample["input"])
                counterfactuals.append(sample["counterfactual_inputs"])
        
        # Create and return a Hugging Face Dataset with the two fields.
        return Dataset.from_dict({
            "input": inputs,
            "counterfactual_inputs": counterfactuals
        })
    
    def load_hf_dataset(self, dataset_path, split, hf_token=None, size=None, name=None, parse_fn=None, ignore_names=[], shuffle=False):
        """
        Load a HuggingFace dataset and reformat it to be compatible with the Task object.
        
        Parameters:
            dataset_path (str): The path or name of the HF dataset
            split (str): "train", "test", or "validation"
            hf_token (str): HuggingFace token
            name (str, optional): Sub-configuration name for the dataset (if any)
            parse_fn (callable, optional): 
                A function that takes a single row from a dataset and returns a string or dict
                to be placed in the "input" column. This is where we do dataset-specific parsing 
                (e.g. extracting digits for arithmetic). If None, we default to using `row["prompt"]`
                or `row["question"]`
        """
        base_dataset = load_dataset(dataset_path, name, split=split, token=hf_token)
        # Will remove later (I will create separate subsets for 2_digit and 1_digit)
        #shuffle the base dataset
        if shuffle:
            base_dataset = base_dataset.shuffle(seed=42)
        if "arithmetic" in dataset_path:
            base_dataset = base_dataset.filter(lambda example: example["num_digit"] == 2)
        
        if size != None:
            if size > len(base_dataset):
                size = len(base_dataset)
            base_dataset = base_dataset.select(range(size))
            
        # Retrieve all counterfactual names
        sample = base_dataset[0]
        counterfactual_names = [k for k in sample.keys() if k.endswith('_counterfactual') and not any(name in k for name in ignore_names)]
        data_dict = {
            counterfactual_name: {"input": [], "counterfactual_inputs": []}
            for counterfactual_name in counterfactual_names
        }
        for row in base_dataset:
            if parse_fn is not None:
                # parse_fn is something like parse_arithmetic_example(row) => returns a string or dict
                input_obj = parse_fn(row) 
            else:
                print("Not able to parse input.")
                input_obj = row.get("question", row.get("prompt", ""))

            for counterfactual_name in counterfactual_names:
                if counterfactual_name in row:
                    cf_data = row[counterfactual_name] 
                else:
                    cf_data = []
                
                data_dict[counterfactual_name]["input"].append(input_obj)
                counterfactual_obj = [parse_fn(cf_data)] # assume it is a list of dict
                data_dict[counterfactual_name]["counterfactual_inputs"].append(counterfactual_obj)

        datasets = {}
        for counterfactual_name in data_dict:
            try:
                name = counterfactual_name.replace("_counterfactual", "_" + split)
                datasets[name] = Dataset.from_dict(data_dict[counterfactual_name])
            except Exception as e:
                print(f"Error creating dataset for {counterfactual_name}: {e} {type(data_dict[counterfactual_name])} {data_dict[counterfactual_name]['input'][0]} {data_dict[counterfactual_name]['counterfactual_inputs'][0]} {split}")
                assert False


        return datasets


# FUNCTIONS FOR PRINTING OUT THE MODEL AND SETTINGS

    def print_structure(self, font=12, node_size=1000):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                (parent, child)
                for child in self.variables
                for parent in self.parents[child]
            ]
        )
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(G, with_labels=True, node_color="green", pos=self.print_pos, font_size=font, node_size=node_size)
        plt.show()

    def print_setting(self, total_setting, font=12, node_size=1000, var_names=False):
        relabeler = {
            var: var + ":\n " + str(total_setting[var]) for var in self.variables
        }
        G = nx.DiGraph()
        G.add_edges_from(
            [
                (parent, child)
                for child in self.variables
                for parent in self.parents[child]
            ]
        )
        plt.figure(figsize=(10, 10))
        G = nx.relabel_nodes(G, relabeler)
        newpos = dict()
        if self.print_pos is not None:
            for var in self.print_pos:
                newpos[relabeler[var]] = self.print_pos[var]
        nx.draw_networkx(G, with_labels=True, node_color="green", pos=newpos, font_size=font, node_size=node_size)
        plt.show()

    
    #=====comprehensive style system for nodes and edges with custom intervention colors=======*
  
    ## Helper functions to analyze tree layout
    
    def _find_root_nodes(self):
        """
        Identify nodes that have no parents. These will be used as 'roots'
        in the breadthfirst layout for hierarchical display.
        """
        root_nodes = []
        for var in self.variables:
            if not self.parents.get(var, []):
                root_nodes.append(var)
        return root_nodes
    

    def _build_outnode_map(self):
        """Create a dictionary mapping nodes to nodes they affect in one pass"""
        # Initialize with empty lists only for nodes that appear as parents
        outnodes = {}
        
        # Single pass through parents dictionary
        for child, parents in self.parents.items():
            for parent in parents:
                # Create list for parent if we haven't seen it
                if parent not in outnodes:
                    outnodes[parent] = [child]
                else:
                    outnodes[parent].append(child)
        
        # Add any remaining nodes that don't affect others (like output node)
        for node in self.variables:
            if node not in outnodes:
                outnodes[node] = []
                
        return outnodes
    
        
    def _find_output_node(self):
        """Find the node that isn't a parent to any other node (has no outnodes)"""
        outnodes = self._build_outnode_map()
        
        # Find node with empty outnode list (doesn't affect any other nodes)
        output_candidates = [node for node, affects in outnodes.items() if not affects]
        
        # There should be exactly one output node
        if len(output_candidates) != 1:
            raise ValueError(f"Found {len(output_candidates)} output nodes, expected exactly 1")
        
        return output_candidates[0]
    
    
    def _find_path_to_output(self, intervention_node):
        """Find path from intervention node to output node using BFS"""
        outnodes = self._build_outnode_map()
        visited = set()
        queue = [(intervention_node, [intervention_node])]
        target_paths = []
        output_node = self._find_output_node()
        
        while queue:
            (current, path) = queue.pop(0)
            
            if current == output_node:
                target_paths.append(path)
                continue
                
            # Simply check outnodes map for next nodes
            for next_node in outnodes[current]:
                if next_node not in visited:
                    visited.add(next_node)
                    new_path = path + [next_node]
                    queue.append((next_node, new_path))
        
        return target_paths
        
    def print_structure_icytoscape(self, font=12, node_size=25):
        try:
            G = nx.DiGraph()
            for child in self.variables:
                G.add_node(child, label=str(child), size=node_size)
                for parent in self.parents.get(child, []):
                    G.add_node(parent, label=str(parent), size=node_size)
                    G.add_edge(parent, child)

            cyto = CytoscapeWidget()
            cyto.graph.add_graph_from_networkx(G)
            
            # Enhanced styling
            cyto.set_style([
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': f'{font}px',
                        'width': f'{node_size}px',
                        'height': f'{node_size}px',
                        'background-color': '#95a5a6',
                        'border-width': 2,
                        'border-color': '#34495e'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'curve-style': 'unbundled-bezier',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 1.5,
                        'line-color': '#7f8c8d',
                        'width': 2
                    }
                }
            ])

            root_nodes = self._find_root_nodes()
            roots_str = str([str(r) for r in root_nodes])
            
            # Simplified layout
            cyto.set_layout(
                name="dagre",
                directed=True,
                rankDir='BT',
                nodeSep=100,
                rankSep=150,
                boundingBox={
                    'x1': 0,
                    'y1': 0,
                    'w': 1500,
                    'h': 800
                }
            )

            logging.info(f"Structure: Created {len(G.nodes)} nodes and {len(G.edges)} edges.")
            display(cyto)
            return cyto
        except Exception as e:
            logging.exception("Error in print_structure_icytoscape:")
            raise

   
    def display_causal_model(self, total_setting, font=12, node_size=25, intervention=None, intervention_color="#5E60CE"):
       """
       Visualizes causal model as interactive directed graph. Shows node values and optional intervention paths.
    
       Parameters:
           values (dict): Node name to value mappings
           font (int): Node label font size, default 12
           node_size (int): Node diameter in pixels, default 25  
           intervention (dict): Single intervention node and value, default None
           intervention_color (str): Hex color for intervention path, default "#5E60CE"
    
       Returns:
           CytoscapeWidget: Interactive graph with click-to-reveal values, variable/value toggle, 
           and intervention highlighting if specified.
       """
       try:
            from ipywidgets import ToggleButton, VBox, HBox, Output, Layout
            
            # create output area for click values
            click_output = Output(layout=Layout(mid_width='300px', width = '50%', margin='10px', overflow='auto'))
            
            # toggle button with specific layout
            toggle = ToggleButton(
                value=False,
                description='Show Values',
                tooltip='Toggle between node names and values',
                layout=Layout(margin='10px')
            )

            # container for both elements with justified spacing
            controls = HBox([
                click_output,  # Left side
                toggle        # Right side
            ], layout=Layout(
                width='100%', 
                justify_content='space-between',
                margin='0 0 10px 0'
            ))

            # Click handler
            def on_click(node):
                with click_output:
                    click_output.clear_output()
                    node_label = node['data']['label']
                    node_value = node['data']['value']
                    print(f"{node_label}: {node_value}")

            # Toggle handler
            def on_toggle_change(change):
                show_values = change['new']
                
                if show_values: 
                    toggle.description = 'Show Variables'
                    cyto.set_style([{
                        'selector': 'node',
                        'style': {
                            'label': 'data(value)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': f'{font}px',
                            'width': f'{node_size}px',
                            'height': f'{node_size}px',
                            'background-color': '#95a5a6',
                            'border-width': 2,
                            'border-color': '#34495e',
                            'font-family': 'Helvetica'
                        }
                    }] + styles[1:])
                else:
                    toggle.description = 'Show Values'
                    cyto.set_style(styles)

            #  Graph creation
            G = nx.DiGraph()
            for var in self.variables:
                label = str(var)
                value = total_setting.get(var, '')
                G.add_node(var, label=label, size=node_size, value=value)
                for parent in self.parents.get(var, []):
                    if parent not in G:
                        parent_value = total_setting.get(parent, '')
                        G.add_node(parent, label=str(parent), size=node_size, value=parent_value)
                    G.add_edge(parent, var)

            cyto = CytoscapeWidget(monitor=True)
            cyto.graph.add_graph_from_networkx(G)
            
            # Add the click handler
            cyto.on('node', 'click', on_click)
            
            # Add the toggle handler
            toggle.observe(on_toggle_change, 'value')

            styles = [
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': f'{font}px',
                        'width': f'{node_size}px',
                        'height': f'{node_size}px',
                        'background-color': '#95a5a6',
                        'border-width': 2,
                        'border-color': '#34495e',
                        'font-family': 'Helvetica'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 1.5,
                        'line-color': '#7f8c8d',
                        'width': 2
                    }
                }
            ]

            # Intervention Styling
            if intervention:
                for node in intervention.keys():
                    styles.append({
                        'selector': f'node[label = "{node}"]',
                        'style': {
                            'background-color': intervention_color,
                            'border-color': intervention_color,
                            'border-width': 3
                        }
                    })
                    
                    paths = self._find_path_to_output(node)
                    
                    for path in paths:
                        for path_node in path:
                            if path_node != node:
                                styles.append({
                                    'selector': f'node[label = "{path_node}"]',
                                    'style': {
                                        'background-color': self._lighten_hex_color(intervention_color),
                                        'border-color': intervention_color,
                                        'border-width': 2
                                    }
                                })
                        
                        for i in range(len(path)-1):
                            styles.append({
                                'selector': f'edge[source = "{path[i]}"][target = "{path[i+1]}"]',
                                'style': {
                                    'curve-style': 'bezier',
                                    'target-arrow-shape': 'triangle',
                                    'arrow-scale': 1.5,
                                    'line-color': intervention_color,
                                    'target-arrow-color': intervention_color,
                                    'width': 3
                                }
                            })

            cyto.set_style(styles)
            
            cyto.set_layout(
                name="dagre",
                directed=True,
                rankDir='BT',
                nodeSep=100,
                rankSep=150,
                boundingBox={
                    'x1': 0,
                    'y1': 0,
                    'w': 1500,
                    'h': 800
                }
            )

            display(controls)
            display(cyto)
            return cyto
       except Exception as e:
            logging.exception("Error in print_setting_icytoscape:")
            raise


    
    # FILTERS AND SAMPLERS

    def generate_equiv_classes(self):
        for var in self.variables:
            if var in self.inputs or var in self.equiv_classes:
                continue
            self.equiv_classes[var] = {val: [] for val in self.values[var]}
            for parent_values in itertools.product(
                *[self.values[par] for par in self.parents[var]]
            ):
                value = self.mechanisms[var](*parent_values)
                self.equiv_classes[var][value].append(
                    {par: parent_values[i] for i, par in enumerate(self.parents[var])}
                )

    def find_live_paths(self, intervention):
        actual_setting = self.run_forward(intervention)
        paths = {1: [[variable] for variable in self.variables]}
        step = 2
        while True:
            paths[step] = []
            for path in paths[step - 1]:
                for child in self.children[path[-1]]:
                    actual_cause = False
                    for value in self.values[path[-1]]:
                        newintervention = copy.deepcopy(intervention)
                        newintervention[path[-1]] = value
                        counterfactual_setting = self.run_forward(newintervention)
                        if counterfactual_setting[child] != actual_setting[child]:
                            actual_cause = True
                    if actual_cause:
                        paths[step].append(copy.deepcopy(path) + [child])
            if len(paths[step]) == 0:
                break
            step += 1
        del paths[1]
        return paths

    def sample_input_tree_balanced(self, output_var=None, output_var_value=None):
        assert output_var is not None or len(self.outputs) == 1
        self.generate_equiv_classes()

        if output_var is None:
            output_var = self.outputs[0]
        if output_var_value is None:
            output_var_value = random.choice(self.values[output_var])


        def create_input(var, value, input={}):
            parent_values = random.choice(self.equiv_classes[var][value])
            for parent in parent_values:
                if parent in self.inputs:
                    input[parent] = parent_values[parent]
                else:
                    create_input(parent, parent_values[parent], input)
            return input

        input_setting = create_input(output_var, output_var_value)
        for input_var in self.inputs:
            if input_var not in input_setting:
                input_setting[input_var] = random.choice(self.values[input_var])
        return input_setting

    def get_path_maxlen_filter(self, lengths):
        def check_path(total_setting):
            input = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input)
            m = max([l for l in paths.keys() if len(paths[l]) != 0])
            if m in lengths:
                return True
            return False

        return check_path

    def get_partial_filter(self, partial_setting):
        def compare(total_setting):
            for var in partial_setting:
                if total_setting[var] != partial_setting[var]:
                    return False
            return True

        return compare

    def get_specific_path_filter(self, start, end):
        def check_path(total_setting):
            input = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input)
            for k in paths:
                for path in paths[k]:
                    if path[0] == start and path[-1] == end:
                        return True
            return False

        return check_path


def simple_example():
    variables = ["A", "B", "C"]
    values = {variable: [True, False] for variable in variables}
    parents = {"A": [], "B": [], "C": ["A", "B"]}

    def A():
        return True

    def B():
        return False

    def C(a, b):
        return a and b

    functions = {"A": A, "B": B, "C": C}
    model = CausalModel(variables, values, parents, functions)
    model.print_structure()
    print("No intervention:\n", model.run_forward(), "\n")
    model.print_setting(model.run_forward())
    print(
        "Intervention setting A and B to TRUE:\n",
        model.run_forward({"A": True, "B": True}),
    )
    print("Timesteps:", model.timesteps)



if __name__ == "__main__":
    simple_example()
