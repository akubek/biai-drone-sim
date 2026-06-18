import pickle
import neat
import graphviz

def draw_net(config, genome, filename="best_network_layered"):
    """Rysuje graf sieci neuronowej NEAT w ułożeniu warstwowym (Lewa -> Prawa)."""
    num_inputs = len(config.genome_config.input_keys)
    is_cascade = (num_inputs == 16)
    # 1. Twój słownik z nazwami (indeksy wejść w neat-python są ujemne)
    # Upewnij się, że kolejność zgadza się z Twoją tablicą state_inputs!
    node_names = {
        # 8 Skanerów (Sensory odległości)
        -1: 'Sensor_angle_0',
        -2: 'Sensor_angle_45',
        -3: 'Sensor_angle_90',
        -4: 'Sensor_angle_135',
        -5: 'Sensor_angle_180',
        -6: 'Sensor_angle_225',
        -7: 'Sensor_angle_270',
        -8: 'Sensor_angle_315',
        
        # Parametry drona i wektory
        -9: 'Vel_X',
        -10: 'Vel_Y',
        -11: 'Angular_Vel',
        -12: 'Sin_Angle',
        -13: 'Cos_Angle',
        -14: 'Norm_Dist',
        -15: 'Sin_Target',
        -16: 'Cos_Target',
        
        # Specyficzne dla End-to-End (jeśli są w grafie)
        -17: 'Actual_L_Thrust',
        -18: 'Actual_R_Thrust',
        
    }

    if is_cascade:
        node_names[0] = 'Joystick_X'
        node_names[1] = 'Joystick_Y'
    else:
        node_names[0] = 'Left_Thrust_Cmd'
        node_names[1] = 'Right_Thrust_Cmd'
    
    # Inicjalizacja grafu: rankdir='LR' wymusza rysowanie od lewej do prawej
    dot = graphviz.Digraph(format='png', node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Arial'})
    dot.attr(rankdir='LR', splines='true', nodesep='0.6', ranksep='2.0')
    
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    
    # 1. WARSTWA WEJŚCIOWA (Nie mają biasu, to tylko czujniki)
    s_in = graphviz.Digraph(name='inputs')
    s_in.attr(rank='source')
    for n in inputs:
        label = node_names.get(n, f"In_{n}")
        s_in.node(str(n), label, style='filled', fillcolor='lightgray')
    dot.subgraph(s_in)

    # 2. WARSTWA WYJŚCIOWA (Mają bias!)
    s_out = graphviz.Digraph(name='outputs')
    s_out.attr(rank='sink')
    for n in outputs:
        base_name = node_names.get(n, f"Out_{n}")
        # Pobieramy bias i formatujemy do 2 miejsc po przecinku
        bias = genome.nodes[n].bias
        label = f"{base_name}\n[Bias: {bias:+.2f}]" 
        s_out.node(str(n), label, style='filled', fillcolor='lightblue')
    dot.subgraph(s_out)

    # 3. WĘZŁY UKRYTE (Też mają bias!)
    hidden_nodes = [n for n in genome.nodes.keys() if n not in outputs]
    for n in hidden_nodes:
        bias = genome.nodes[n].bias
        label = f"N_{n}\n[b: {bias:+.2f}]"
        dot.node(str(n), label, shape='circle', fillcolor='white')

    # 4. POŁĄCZENIA (Wagi na liniach)
    for cg in genome.connections.values():
        if cg.enabled:
            color = '#2E8B57' if cg.weight > 0 else '#CD5C5C'
            width = str(0.2 + abs(cg.weight / 3.0))
            
            in_node = cg.key[0]
            out_node = cg.key[1]
            
            # Formatujemy wagę z plusem/minusem
            weight_label = f"{cg.weight:+.2f}"
            
            dot.edge(
                str(in_node), 
                str(out_node), 
                color=color, 
                penwidth=width,
                label=weight_label,     # Etykieta na linii
                fontcolor=color,        # Kolor tekstu taki sam jak linii
                fontsize='10',          # Nieco mniejsza czcionka, żeby nie zasłaniała
                fontname='Arial'
            )

    dot.render(filename, view=True)

# Użycie:
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, "conf/neat-cascade.txt")

with open("models/best_drone_cascade.pkl", "rb") as f:
    winner = pickle.load(f)

draw_net(config, winner)