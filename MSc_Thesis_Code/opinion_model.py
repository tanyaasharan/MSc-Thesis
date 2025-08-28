import random

# increment n iterations for traders that communicated
def inc_n_iter(traders):
    for trader in traders:
        trader.n_iter += 1


def relative_disagreement_step(weight_ra, weight_rd, prob, traders, graph, lob):
    """
    Performs a step of the relative disagreement opinion dynamics model.

    Args:
        weight_ra (float): The weight to apply during relative agreement updates.
        weight_rd (float): The weight to apply during relative disagreement updates.
        prob (float): The probability of triggering a disagreement update.
        traders (dict): A dictionary of trader objects, indexed by trader ID.
        graph (networkx.Graph): The social graph representing trader connections.
        lob (dict): The current state of the Limit Order Book.
    """
    if graph is None:
        # If no graph, pick two random traders
        i, j = random.sample(list(traders.keys()), 2)
    else:
        # If graph exists, pick a random trader and one of their neighbors
        i_node = random.sample(list(graph.nodes()), 1)[0]
        neighbors = list(graph.neighbors(i_node))
        if not neighbors: # Handle nodes with no neighbors
            return # No interaction if no neighbors
        j_node = random.sample(neighbors, 1)[0]
        i = graph.nodes[i_node].get('t_id', i_node) # Get trader ID from node attribute or use node ID
        j = graph.nodes[j_node].get('t_id', j_node) # Get trader ID from node attribute or use node ID


    # Ensure i and j are strings if trader IDs are strings
    i = str(i)
    j = str(j)

    # Ensure traders exist in the traders dictionary
    if i not in traders or j not in traders:
        print(f"Warning: Trader ID {i} or {j} not found in traders dictionary.")
        return


    X_i = traders[i].opinion  # trader i opinion value
    u_i = traders[i].uncertainty  # trader i uncertainty value
    X_j = traders[j].opinion  # trader j opinion value
    u_j = traders[j].uncertainty  # trader j uncertainty value

    # Calculates overlap between trader opinion uncertainties
    h_ij = min((X_i + u_i), (X_j + u_j)) - max((X_i - u_i), (X_j - u_j))
    h_ji = min((X_j + u_j), (X_i + u_i)) - max((X_j - u_j), (X_i - u_i))
    # Calculate overlap disagreement
    g_ij = max((X_i - u_i), (X_j - u_j)) - min((X_i + u_i), (X_j + u_j))
    g_ji = max((X_j - u_j), (X_i - u_i)) - min((X_j + u_j), (X_i + u_i))

    # Update with probability prob (for disagreement)
    if random.random() <= prob:
        if (g_ji > u_j):
            RD_ji = (g_ji / u_j) - 1
            # Use negative weight_rd for disagreement
            traders[i].set_opinion(lob, -weight_rd, prob, RD_ji, X_j)
            traders[i].set_uncertainty(u_i + (-weight_rd * RD_ji * (u_j - u_i)))
        if (g_ij > u_i):
            RD_ij = (g_ij / u_i) - 1
            # Use negative weight_rd for disagreement
            traders[j].set_opinion(lob, -weight_rd, prob, RD_ij, X_i)
            traders[j].set_uncertainty(u_j + (-weight_rd * RD_ij * (u_i - u_j)))
    # Agreement update (always happens if there's overlap)
    if (h_ji > u_j):
        RA_ji = (h_ji / u_j) - 1
        # Use weight_ra for agreement
        traders[i].set_opinion(lob, weight_ra, prob, RA_ji, X_j)
        traders[i].set_uncertainty(u_i + (weight_ra * RA_ji * (u_j - u_i)))
    if (h_ij > u_i):
        RA_ij = (h_ij / u_i) - 1
        # Use weight_ra for agreement
        traders[j].set_opinion(lob, weight_ra, prob, RA_ij, X_i)
        traders[j].set_uncertainty(u_j + (weight_ra * RA_ij * (u_i - u_j)))

    inc_n_iter([traders[i], traders[j]])


def external_opinion_step(X_i, u_i, X_e, n_e, prob):
    h_ei = min((X_e + n_e), (X_i + u_i)) - max((X_e - n_e), (X_i - u_i))
    g_ei = max((X_e - n_e), (X_i - u_i)) - min((X_e + n_e), (X_i + u_i))

    if random.random() <= prob:
        if (g_ei > n_e) :
            RD_ei = (g_ei / n_e) - 1
            weight_e = -1
            return weight_e, RD_ei
    if (h_ei > n_e) :
        RA_ei = (h_ei / n_e) - 1
        weight_e = 1
        return weight_e, RA_ei
    else:
        return 0, 0