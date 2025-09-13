"""
DOES Model: Dynamics of Opinion and Emotion in Social Media

This module implements the core dynamics of the DOES model, which simulates
the coupled evolution of opinions and emotions in social media.

The model consists of:
1. Opinion dynamics layer: Opinion formation and change
2. Emotion dynamics layer: Emotion synchronization using Kuramoto model
3. Cross-layer coupling: Opinion influences emotion and vice versa
"""

import numpy as np
import networkx as nx
import random
from tqdm import tqdm


def opinion_init(graph, mode):
    """
    Initialize opinions or emotions on the network nodes.
    - mode: 1 for opinion initialization, 2 for emotion initialization
    """
    if mode == 1:
        for node in graph.nodes():
            graph.nodes[node]['opi'] = round(random.uniform(0, 1), 2)

    if mode == 2:
        for node in graph.nodes():
            graph.nodes[node]['emo'] = round(random.uniform(-np.pi, np.pi), 2)  # Emotion phase
            graph.nodes[node]['freq'] = round(random.uniform(-0.5, 0.5), 2)    # Natural frequency

    return graph


def cal_local_order_parameter(graph, neighbors):
    """
    Calculate the local order parameter for emotion dynamics model.
    """
    r = 0
    ki = len(neighbors)

    if ki:
        r = abs(1 / ki * np.sum([np.exp(1j * graph.nodes[j]['emo']) for j in neighbors]))

    return r


def cal_ksi(opinion_graph, emotion_graph, node, alpha):
    """
    Calculate the coupling strength from emotion to opinion dynamics.
    """
    neighbors = list(opinion_graph.neighbors(node))
    emotion_i = emotion_graph.nodes[node]['emo']
    r = cal_local_order_parameter(emotion_graph, neighbors)
    
    return r / (1 + np.exp(-alpha * emotion_i))


def opinion_dynamic(opinion_graph, emotion_graph, lamb, alpha):
    """
    Calculate opinion dynamics update for all nodes.
    
    Returns:
    - dp: Dictionary of opinion changes for each node
    """
    dp = {}
    
    for node in opinion_graph.nodes:
        neighbors = list(opinion_graph.neighbors(node))

        if neighbors:
            ksi = cal_ksi(opinion_graph, emotion_graph, node, alpha)
            dp[node] = lamb * ksi * np.sum([
                (opinion_graph.nodes[j]['opi'] - opinion_graph.nodes[node]['opi']) 
                for j in neighbors
            ])
        else:
            dp[node] = 0

    return dp


def cal_distance(graph, node, neighbors):
    """
    Calculate opinion distance for emotion dynamics coupling.
    
    Returns:
    - dis: Normalized opinion distance
    """
    opi_i = graph.nodes[node]['opi']
    distances = []
    
    for j in neighbors:
        opi_j = graph.nodes[j]['opi']
        distance = (opi_j - opi_i) ** 2
        distances.append(distance)

    # Sum the distances and normalize
    sum_distances = np.sum(distances)
    ki = len(neighbors)
    dis = np.sqrt(sum_distances / ki)

    return dis


def cal_eta(opinion_graph, emotion_graph, node, beta):
    """
    Calculate the coupling strength from opinion to emotion dynamics.
    """
    neighbors = list(emotion_graph.neighbors(node))
    dis = cal_distance(opinion_graph, node, neighbors)

    return np.exp(-beta * dis ** 2)


def emotion_dynamic(opinion_graph, emotion_graph, mu, beta):
    """
    Calculate emotion dynamics update for all nodes.
    
    Returns:
    - dp: Dictionary of emotion phase changes for each node
    """
    dp = {}
    
    for node in emotion_graph.nodes:
        neighbors = list(emotion_graph.neighbors(node))

        # Natural frequency
        dp[node] = emotion_graph.nodes[node]['freq']

        if neighbors:
            eta = cal_eta(opinion_graph, emotion_graph, node, beta)
            dp[node] += mu * eta * np.sum([
                np.sin(emotion_graph.nodes[j]['emo'] - emotion_graph.nodes[node]['emo']) 
                for j in neighbors
            ])

    return dp


def runge_kutta(opinion_graph, emotion_graph, mu, beta, dt):
    """
    4th order Runge-Kutta method for emotion dynamics.
    
    Returns:
    - dp: Dictionary of emotion phase changes for each node
    """
    dp = {}

    k1 = emotion_dynamic(opinion_graph, emotion_graph, mu, beta)

    # k2 calculation
    emotion_temp = emotion_graph.copy()
    for node in emotion_temp.nodes:
        emotion_temp.nodes[node]['emo'] += k1[node] * dt / 2
        emotion_temp.nodes[node]['emo'] = np.mod(
            emotion_temp.nodes[node]['emo'] + np.pi, 2 * np.pi
        ) - np.pi
    k2 = emotion_dynamic(opinion_graph, emotion_temp, mu, beta)

    # k3 calculation
    emotion_temp = emotion_graph.copy()
    for node in emotion_temp.nodes:
        emotion_temp.nodes[node]['emo'] += k2[node] * dt / 2
        emotion_temp.nodes[node]['emo'] = np.mod(
            emotion_temp.nodes[node]['emo'] + np.pi, 2 * np.pi
        ) - np.pi
    k3 = emotion_dynamic(opinion_graph, emotion_temp, mu, beta)

    # k4 calculation
    emotion_temp = emotion_graph.copy()
    for node in emotion_temp.nodes:
        emotion_temp.nodes[node]['emo'] += k3[node] * dt
        emotion_temp.nodes[node]['emo'] = np.mod(
            emotion_temp.nodes[node]['emo'] + np.pi, 2 * np.pi
        ) - np.pi
    k4 = emotion_dynamic(opinion_graph, emotion_temp, mu, beta)

    # Final update
    for node in emotion_graph.nodes:
        dp[node] = (k1[node] + 2*k2[node] + 2*k3[node] + k4[node]) / 6
        
    return dp


def run_simulation(opinion_graph, emotion_graph, T, dt, lamb, alpha, mu, beta, 
                  flag=False, tolerance=1e-5, stability_threshold=5):
    """
    Run the coupled DOES model simulation.
    
    Parameters:
    - opinion_graph: Initial opinion graph
    - emotion_graph: Initial emotion graph
    - T: Total simulation time
    - dt: Time step
    - lamb: Opinion dynamics global susceptibility parameter
    - alpha: Emotionality parameter
    - mu: Emotion dynamics empathy parameter
    - beta: Empathic selectivity parameter
    - flag: Stopping criterion (1: both stable, 2: opinion stable, other: run full time)
    - tolerance: Stability tolerance
    - stability_threshold: Number of consecutive stable steps required
    
    Returns:
    - opinion_results: Dictionary with time series of opinion values
    - emotion_results: Dictionary with time series of emotion values
    """
    # Determine saving interval based on network size
    n_nodes = len(opinion_graph.nodes)
    save_interval = 50 if n_nodes > 10000 else 1

    # Initialize storage dictionaries
    opinion_results = {
        'time_steps': [], 
        'opi_values': {node: [] for node in opinion_graph.nodes}
    }
    emotion_results = {
        'time_steps': [], 
        'emo_values': {node: [] for node in emotion_graph.nodes}
    }

    # Store previous states for stability checking
    opinion_prev = opinion_graph.copy()
    emotion_prev = emotion_graph.copy()

    t = 0
    step = 0
    change_counter = 0
    pbar = tqdm(total=T / dt, desc="DOES Simulation")

    while t < T:
        # Save data at specified intervals
        if step % save_interval == 0:
            opinion_results['time_steps'].append(step)
            emotion_results['time_steps'].append(step)
            
            for node in opinion_graph.nodes:
                opinion_results['opi_values'][node].append(opinion_graph.nodes[node]['opi'])
            for node in emotion_graph.nodes:
                emotion_results['emo_values'][node].append(emotion_graph.nodes[node]['emo'])

        # Calculate dynamics
        dp_opinion = opinion_dynamic(opinion_graph, emotion_graph, lamb, alpha)
        dp_emotion = runge_kutta(opinion_graph, emotion_graph, mu, beta, dt)

        # Update opinions
        for node in opinion_graph.nodes:
            opinion_graph.nodes[node]['opi'] += dp_opinion[node] * dt

        # Update emotions
        for node in emotion_graph.nodes:
            emotion_graph.nodes[node]['emo'] += dp_emotion[node] * dt
            emotion_graph.nodes[node]['emo'] = np.mod(
                emotion_graph.nodes[node]['emo'] + np.pi, 2 * np.pi
            ) - np.pi
            
            # Check for NaN values
            if np.isnan(emotion_graph.nodes[node]['emo']):
                print("Warning: NaN detected in emotion values")
                break

        # Check for stability (after a few initial steps)
        if step > 2:
            # Calculate opinion change
            opi_change = np.mean([
                abs(opinion_graph.nodes[node]['opi'] - opinion_prev.nodes[node]['opi']) 
                for node in opinion_graph.nodes()
            ])
            
            # Calculate emotion phase difference change
            phase_diff_changes = []
            for node in emotion_graph.nodes:
                neighbors = list(emotion_graph.neighbors(node))
                if neighbors:
                    diff_change = np.mean([
                        abs((emotion_graph.nodes[node]['emo'] - emotion_graph.nodes[nbr]['emo']) -
                            (emotion_prev.nodes[node]['emo'] - emotion_prev.nodes[nbr]['emo'])) 
                        for nbr in neighbors
                    ])
                    phase_diff_changes.append(diff_change)
            phase_diff_change = np.mean(phase_diff_changes) if phase_diff_changes else 0
            
        else:
            opi_change = 1
            phase_diff_change = 1

        # Check stopping criteria
        if flag == 1:  # Both opinion and emotion stability
            if opi_change < tolerance and phase_diff_change < tolerance:
                change_counter += 1
            else:
                change_counter = 0
        elif flag == 2:  # Only opinion stability
            if opi_change < tolerance:
                change_counter += 1
            else:
                change_counter = 0
        else:
            change_counter = 0

        # Stop if stable for enough steps
        if change_counter >= stability_threshold:
            print(f"Convergence reached at step {step}")
            break

        # Update previous states
        opinion_prev = opinion_graph.copy()
        emotion_prev = emotion_graph.copy()

        t += dt
        step += 1
        pbar.update(1)

    pbar.close()
    
    return opinion_results, emotion_results
