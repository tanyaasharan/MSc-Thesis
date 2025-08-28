def klemm_eguiluz_graph(num_nodes, m=3, mu=0.1, seed=42):
    import networkx as nx
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.Graph()
    active_nodes = list(range(m))  # start with m fully connected nodes
    G.add_nodes_from(active_nodes)
    for i in active_nodes:
        for j in active_nodes:
            if i < j:
                G.add_edge(i, j)
    
    for new_node in range(m, num_nodes):
        G.add_node(new_node)
        for _ in range(m):
            if random.random() < mu:
                # connect to a random node (preferential attachment)
                degrees = np.array([G.degree(n) for n in G.nodes()])
                probs = degrees / degrees.sum()
                target = np.random.choice(list(G.nodes()), p=probs)
            else:
                # connect to a random active node
                target = random.choice(active_nodes)
            G.add_edge(new_node, target)
        
        active_nodes.append(new_node)
        # Deactivate one old active node (low degree more likely to be deactivated)
        weights = [1 / G.degree(n) for n in active_nodes]
        total = sum(weights)
        probs = [w / total for w in weights]
        deactivated = random.choices(active_nodes, weights=probs, k=1)[0]
        active_nodes.remove(deactivated)

    return G
