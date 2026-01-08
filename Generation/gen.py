import random
import numpy as np
import json

# Style presets for CircuitikZ symbols (used to diversify synthetic rendering)
RESISTOR_STYLES = [
    "R",                  # default zigzag resistor
    "american resistor",  # rectangular resistor
    "european resistor",  # IEC-style resistor
]

CAPACITOR_STYLES = [
    "C",                  # regular capacitor
    "polar capacitor",    # polarized capacitor
]

INDUCTOR_STYLES = [
    "american inductor",
    "cute inductor",
    "european inductor",
]

# YOLO class id for the "Text_Label" category (kept as string to match write format)
TEXT_CLASS_ID = "11"


def generate_random_nets(num_nets, grid_size):
    """
    Randomly place net anchor points on a discrete grid.
    These anchor points later serve as connection targets for device terminals.
    """
    nets = []
    for i in range(num_nets):
        nx = random.randint(0, grid_size - 1)
        ny = random.randint(0, grid_size - 1)
        nets.append({"name": f"N{i}", "position": (nx, ny)})
    return nets


def find_compatible_nets_for_device(device_x, device_y, nets):
    """
    Pick nets that are reasonably close to a device location so that
    wires are not too short (degenerate) or too long (messy).
    """
    compatible = []
    for net in nets:
        nx, ny = net["position"]

        # Loose "nearby" constraint: close in x OR close in y
        if abs(nx - device_x) <= 1 or abs(ny - device_y) <= 1:
            manh = abs(nx - device_x) + abs(ny - device_y)

            # Empirical band to keep connection geometry stable
            if 2 <= manh <= 5:
                compatible.append(net)
    return compatible


def create_device(device_type, index, grid_size):
    """
    Create a device template with:
    - a random grid position
    - terminal count (1/2/3)
    - a CircuitikZ nodeType for rendering diversity
    """
    dx = random.randint(0, grid_size - 1)
    dy = random.randint(0, grid_size - 1)

    # Decide terminal count + canonical naming
    if device_type == "Ground":
        t_count = 1
        dev_name = f"GND{index}"
    elif device_type == "Resistor":
        t_count = 2
        dev_name = f"R{index}"
    elif device_type == "Capacitor":
        t_count = 2
        dev_name = f"C{index}"
    elif device_type == "Inducer":
        t_count = 2
        dev_name = f"L{index}"
    elif device_type == "Diode":
        t_count = 2
        dev_name = f"D{index}"
    elif device_type == "Voltage":
        t_count = 2
        dev_name = f"V{index}"
    elif device_type == "Current":
        t_count = 2
        dev_name = f"I{index}"
    elif device_type in ["PMOS", "NMOS", "NPN", "PNP"]:
        t_count = 3
        dev_name = f"{device_type}{index}"
    else:
        t_count = 2
        dev_name = f"{device_type}{index}"

    # Map to CircuitikZ symbol keywords / variants
    if device_type == "PMOS":
        node_type = "pmos"
    elif device_type == "NMOS":
        node_type = "nmos"
    elif device_type == "NPN":
        node_type = "npn"
    elif device_type == "PNP":
        node_type = "pnp"
    elif device_type == "Inducer":
        node_type = random.choice(INDUCTOR_STYLES)
    elif device_type == "Diode":
        node_type = "D"
    elif device_type == "Resistor":
        node_type = random.choices(
            RESISTOR_STYLES,
            weights=[3, 2, 2],
            k=1,
        )[0]
    elif device_type == "Capacitor":
        node_type = random.choices(
            CAPACITOR_STYLES,
            weights=[3, 2],
            k=1,
        )[0]
    elif device_type == "Ground":
        node_type = "ground"
    elif device_type == "Voltage":
        node_type = random.choice(
            ["american voltage source", "european voltage source"]
        )
    elif device_type == "Current":
        node_type = random.choice(
            ["american current source", "european current source"]
        )
    else:
        node_type = "ERROR"

    return {
        "type": device_type,
        "name": dev_name,
        "position": (dx, dy),
        "terminal_count": t_count,
        "nodeType": node_type,
    }


def assign_nets_to_device(device, nets, devices, max_retries=50):
    """
    Assign compatible nets to a device, while also enforcing a simple
    overlap / crowding constraint against already-placed devices.

    Returns:
        device (dict) on success, or None if placement fails after retries.
    """
    global grid_size

    for _ in range(max_retries):
        dx, dy = device["position"]
        compatible_nets = find_compatible_nets_for_device(dx, dy, nets)

        # Need at least terminal_count distinct nets to connect the device
        if len(compatible_nets) >= device["terminal_count"]:
            chosen = random.sample(compatible_nets, k=device["terminal_count"])
            device["nets"] = [net["name"] for net in chosen]

            # Avoid heavy overlaps / overly tight clusters of devices
            needs_reposition = False
            for dev in devices:
                # Exact same grid cell
                if dev["position"] == device["position"]:
                    device["position"] = (
                        random.randint(0, grid_size - 1),
                        random.randint(0, grid_size - 1),
                    )
                    needs_reposition = True

                # Too close (within a 2x2 neighborhood)
                if (
                    abs(dev["position"][0] - device["position"][0]) <= 2
                    and abs(dev["position"][1] - device["position"][1]) <= 2
                ):
                    device["position"] = (
                        random.randint(0, grid_size - 1),
                        random.randint(0, grid_size - 1),
                    )
                    needs_reposition = True

            if needs_reposition:
                continue
            return device

        # Not enough compatible nets -> re-sample position and retry
        device["position"] = (
            random.randint(0, grid_size - 1),
            random.randint(0, grid_size - 1),
        )

    return None


def generate_random_circuit(
    num_nets=8,
    max_resistors=3,
    max_capacitors=3,
    max_pmos=3,
    max_nmos=3,
    max_npn=3,
    max_pnp=3,
    max_inducer=3,
    max_diode=3,
    max_ground=3,
    grid_size_param=5,
):
    """
    Generate a random circuit "layout graph" consisting of:
    - a set of nets (anchor points)
    - a set of devices, each connected to a few nets

    This function does NOT draw anything; it only produces the structured data
    later used for rendering and label export.
    """
    global grid_size
    grid_size = grid_size_param

    nets = generate_random_nets(num_nets, grid_size)

    # Sample device counts (at least 1 for each category due to randint(1, max_*))
    num_res = random.randint(1, max_resistors)
    num_cap = random.randint(1, max_capacitors)
    num_p = random.randint(1, max_pmos)
    num_n = random.randint(1, max_nmos)
    num_npn = random.randint(1, max_npn)
    num_pnp = random.randint(1, max_pnp)
    num_inducer = random.randint(1, max_inducer)
    num_diode = random.randint(1, max_diode)
    num_ground = random.randint(1, max_ground)

    dev_types = (
        ["Resistor"] * num_res
        + ["Capacitor"] * num_cap
        + ["PMOS"] * num_p
        + ["NMOS"] * num_n
        + ["NPN"] * num_npn
        + ["PNP"] * num_pnp
        + ["Inducer"] * num_inducer
        + ["Diode"] * num_diode
        + ["Ground"] * num_ground
    )
    random.shuffle(dev_types)

    counters = {
        "Resistor": 0,
        "Capacitor": 0,
        "PMOS": 0,
        "NMOS": 0,
        "NPN": 0,
        "PNP": 0,
        "Inducer": 0,
        "Diode": 0,
        "Ground": 0,
    }

    devices = []
    for dev_type in dev_types:
        counters[dev_type] += 1
        dev = create_device(dev_type, counters[dev_type], grid_size)
        dev_with_nets = assign_nets_to_device(dev, nets, devices)
        if dev_with_nets is not None:
            devices.append(dev_with_nets)

    # Fallback: ensure we always return at least one device
    if not devices:
        nets = generate_random_nets(4, grid_size)
        dev = create_device("Resistor", 1, grid_size)
        dev_with_nets = assign_nets_to_device(dev, nets, [])
        if dev_with_nets is not None:
            devices.append(dev_with_nets)

    return nets, devices


def generate_text_labels(
    devices,
    nets,
    grid_size,
    max_device_labels=8,
    max_net_labels=8,
    max_free_labels=4,
    min_text_dist=1.0,
    max_retries=15,
):
    """
    Generate synthetic text annotations and treat them as first-class labels.

    Three categories:
      1) Device labels: placed near selected devices (e.g., R1, C2, NMOS3)
      2) Net labels: placed near net anchor points (e.g., N0, N1, ...)
      3) Free annotations: auxiliary notes (e.g., VDD, CLK, bias)

    Placement uses a simple minimum-distance constraint to avoid text clutter.
    """
    text_labels = []

    def too_close(x, y):
        """Check if a candidate position is too close to existing text labels."""
        for t in text_labels:
            dx = x - t["x"]
            dy = y - t["y"]
            if dx * dx + dy * dy < min_text_dist * min_text_dist:
                return True
        return False

    # --- 1) Device labels (only for selected device types) ---
    device_name_candidates = []
    for dev in devices:
        if dev["type"] in ["Resistor", "Capacitor", "PMOS", "NMOS", "Voltage", "Current"]:
            device_name_candidates.append(dev)

    dev_samples = []
    if device_name_candidates:
        n_dev = min(len(device_name_candidates), max_device_labels)
        n_dev = max(n_dev, 1)
        dev_samples = random.sample(device_name_candidates, n_dev)

    for idx, dev in enumerate(dev_samples):
        dx, dy = dev["position"]
        base_offset = random.uniform(0.6, 1.0)

        # Try a few random sides around the device until spacing is acceptable
        lx, ly = dx, dy
        for _ in range(max_retries):
            side = random.choice(["up", "down", "left", "right"])
            if side == "up":
                lx, ly = dx, dy + base_offset
            elif side == "down":
                lx, ly = dx, dy - base_offset
            elif side == "left":
                lx, ly = dx - base_offset, dy
            else:
                lx, ly = dx + base_offset, dy

            # Clamp into the approximate drawing canvas bounds
            lx = max(0, min(grid_size, lx))
            ly = max(-grid_size / 2, min(grid_size / 2, ly))

            if not too_close(lx, ly):
                break

        text_labels.append(
            {"name": f"TXT_DEV{idx}", "text": dev["name"], "x": lx, "y": ly}
        )

    # --- 2) Net labels ---
    net_samples = []
    if nets:
        n_net = min(len(nets), max_net_labels)
        n_net = max(n_net, 1)
        net_samples = random.sample(nets, n_net)

    for idx, net in enumerate(net_samples):
        nx, ny = net["position"]

        lx, ly = nx, ny
        for _ in range(max_retries):
            lx = nx + random.uniform(-0.5, 0.5)
            ly = ny + random.uniform(-0.5, 0.5)
            lx = max(0, min(grid_size, lx))
            ly = max(-grid_size / 2, min(grid_size / 2, ly))
            if not too_close(lx, ly):
                break

        text_labels.append(
            {"name": f"TXT_NET{idx}", "text": net["name"], "x": lx, "y": ly}
        )

    # --- 3) Free annotations ---
    free_words = [
        "VDD", "GND", "CLK", "bias", "out", "in", "Vref", "Ibias",
        "node1", "node2", "Amp", "Stage1",
    ]
    num_free = random.randint(0, max_free_labels)

    for idx in range(num_free):
        # Sample a random location and enforce spacing constraint
        lx = random.uniform(0, grid_size)
        ly = random.uniform(-grid_size / 2, grid_size / 2)

        for _ in range(max_retries):
            lx = random.uniform(0, grid_size)
            ly = random.uniform(-grid_size / 2, grid_size / 2)
            if not too_close(lx, ly):
                break

        text_labels.append(
            {"name": f"TXT_FREE{idx}", "text": random.choice(free_words), "x": lx, "y": ly}
        )

    return text_labels


def build_device_edges(devices, nets):
    """
    Build an undirected "device graph" used as graph ground-truth (GED).
    Two devices are connected if they share a net. If no shared-net edges exist,
    fall back to connecting nearest neighbors to avoid empty graphs.
    """
    net_to_devices = {net["name"]: [] for net in nets}

    for dev in devices:
        for net_name in dev.get("nets", []):
            if net_name in net_to_devices:
                net_to_devices[net_name].append(dev["name"])

    edges = []
    for net_name, dev_list in net_to_devices.items():
        if len(dev_list) >= 2:
            for i in range(len(dev_list)):
                for j in range(i + 1, len(dev_list)):
                    edges.append({"u": dev_list[i], "v": dev_list[j], "via": net_name})

    # Fallback: if nothing shares nets, connect devices by spatial proximity
    if not edges and len(devices) >= 2:
        positions = {d["name"]: d["position"] for d in devices}
        used_pairs = set()

        for d in devices:
            x0, y0 = positions[d["name"]]
            best = None
            best_dist2 = 1e18

            for other in devices:
                if other["name"] == d["name"]:
                    continue
                pair = tuple(sorted((d["name"], other["name"])))
                if pair in used_pairs:
                    continue

                x1, y1 = positions[other["name"]]
                dist2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best = other["name"]

            if best is not None:
                pair = tuple(sorted((d["name"], best)))
                used_pairs.add(pair)
                edges.append({"u": d["name"], "v": best, "via": None})

    return edges


def write_graph_json(json_path, devices, edges):
    """Serialize device graph ground-truth to JSON for downstream GED evaluation."""
    graph = {
        "meta": {"num_devices": len(devices), "num_edges": len(edges)},
        "devices": [
            {
                "id": dev["name"],
                "type": dev["type"],
                "position": [float(dev["position"][0]), float(dev["position"][1])],
            }
            for dev in devices
        ],
        "edges": edges,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


if __name__ == "__main__":
    import sys

    # Base filename stem, e.g., "1" -> produces "1.txt" and "1_graph.json"
    base_name = sys.argv[1]
    label_filename = base_name + ".txt"              # YOLO label output (one line per bbox)
    graph_filename = base_name + "_graph.json"       # device graph ground-truth (GED)

    # Randomize grid size per sample to increase layout diversity
    grid_size = random.randint(5, 10)

    # Generate circuit structure (nets + devices)
    nets, devices = generate_random_circuit(
        num_nets=random.randint(40, 50),
        max_resistors=5,
        max_capacitors=5,
        max_pmos=10,
        max_nmos=10,
        max_npn=5,
        max_pnp=5,
        max_inducer=5,
        max_diode=5,
        max_ground=5,
        grid_size_param=grid_size,
    )

    # Build net -> device adjacency (used to decide whether to draw junction dots)
    net_to_devices = {net["name"]: [] for net in nets}
    for dev in devices:
        for net_name in dev.get("nets", []):
            if net_name in net_to_devices:
                net_to_devices[net_name].append(dev["name"])

    # Toggle whether to draw wires / junction markers
    show_lines = True

    # Export device graph ground-truth
    edges = build_device_edges(devices, nets)
    write_graph_json(graph_filename, devices, edges)

    # Generate synthetic text annotations (also exported as YOLO labels)
    text_labels = generate_text_labels(devices, nets, grid_size)

    # Count how many device terminals are attached to each net (junction heuristic)
    countN = {net["name"]: 0 for net in nets}
    for dev in devices:
        for net_name in dev.get("nets", []):
            if net_name in countN:
                countN[net_name] += 1

    # Randomize line width to diversify rendering style
    lineW = random.choice([0.5, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.5])

    # Emit a full standalone LaTeX document that:
    # 1) draws the schematic via circuitikz
    # 2) writes YOLO labels into label_filename via a TeX file handle
    print(
        r"""
\documentclass{standalone}
\usepackage{circuitikz}
\usepackage{pgfmath}
\usepackage{sansmath}
\begin{document}
\sansmath
% Open a write handle and stream YOLO labels into the output file: """
        + label_filename
        + r"""
\newwrite\posfile
\immediate\openout\posfile="""
        + label_filename
        + r"""
\begin{circuitikz}["""
        + f"line width={lineW-0.3}pt"
        + r"""]
"""
    )
    print(r"\ctikzset{label/align = straight}")

    mynet = {net["name"]: net for net in nets}

    # Draw junction dots for nets that connect many terminals (simple visual cue)
    if show_lines:
        for net in nets:
            if countN.get(net["name"], 0) >= 3:
                nx, ny = net["position"]
                size = lineW + random.randint(1, 2)
                print(f"\\draw[fill=black] ({nx},{ny}) circle ({size}pt);")

    # Draw devices and optionally connect them to their assigned nets
    for dev in devices:
        dev_x, dev_y = dev["position"]

        # Random appearance perturbations
        rotate = random.choice([0, 90, 180, 270])
        x = random.choice([(1, 1), (1, 1.3), (1.3, 1), (1.5, 1.3), (1.3, 1.5)])
        xscale, yscale = x[0], x[1]

        # 1-terminal device (Ground)
        if dev["terminal_count"] == 1:
            # local bounding box is used later to extract bbox coordinates
            print(f"\\begin{{scope}}[local bounding box={dev['name']}BB]")
            print(
                f"\\draw ({dev_x},{dev_y}) node[{dev['type'].lower()},xscale={xscale}, yscale={yscale}]({dev['name']}){{}};"
            )
            print("\\end{scope}")

            if show_lines and dev.get("nets"):
                nx, ny = mynet[dev["nets"][0]]["position"]
                print(f"\\draw ({dev['name']}.center) to ({nx},{ny});")

        # 2-terminal devices (R/C/L/D/V/I)
        if dev["terminal_count"] == 2:
            print(f"\\begin{{scope}}[local bounding box={dev['name']}BB]")

            # Choose a horizontal or vertical orientation
            direction = random.choice(["H", "V"])
            if direction == "H":
                p1x, p1y = dev_x + 1, dev_y
                p2x, p2y = dev_x - 1, dev_y
            else:
                p1x, p1y = dev_x, dev_y + 1
                p2x, p2y = dev_x, dev_y - 1

            node_type = dev["nodeType"]
            if dev["type"] == "Resistor":
                # Randomize zig count for zigzag resistors
                print(r"\ctikzset{resistors/zigs = " + str(random.choice([2, 3, 4])) + "}")
                print(f"\\draw ({p1x},{p1y}) to[{node_type}] ({p2x},{p2y});")
            else:
                print(f"\\draw ({p1x},{p1y}) to[{node_type}] ({p2x},{p2y});")

            print("\\end{scope}")

            if show_lines and len(dev.get("nets", [])) >= 2:
                nx, ny = mynet[dev["nets"][0]]["position"]
                print(f"\\draw ({p1x},{p1y}) to ({nx},{ny});")
                nx, ny = mynet[dev["nets"][1]]["position"]
                print(f"\\draw ({p2x},{p2y}) to ({nx},{ny});")

        # 3-terminal devices (MOSFET/BJT)
        if dev["terminal_count"] == 3:
            print(f"\\begin{{scope}}[local bounding box={dev['name']}BB]")

            if dev["type"] in ["PMOS", "NMOS"]:
                # Randomize MOS symbol decorations (circle/bulk) for appearance diversity
                circle = random.choice([0, 1, 2])
                CIRCLE = "" if circle == 0 else ("emptycircle" if circle == 1 else "nocircle")

                bulk = random.choice([0, 1])
                BULK = "" if bulk == 0 else "bulk"

                print(
                    f"\\draw ({dev_x},{dev_y}) node[{dev['type'].lower()},{BULK}, rotate={rotate}, xscale={xscale}, yscale={yscale},{CIRCLE},arrowmos]({dev['name']}){{}};"
                )
            else:
                print(
                    f"\\draw ({dev_x},{dev_y}) node[{dev['type'].lower()}, rotate={rotate}, xscale={xscale}, yscale={yscale}]({dev['name']}){{}};"
                )

            print("\\end{scope}")

            if show_lines and len(dev.get("nets", [])) >= 3:
                nx, ny = mynet[dev["nets"][0]]["position"]
                print(f"\\draw ({dev['name']}.S) to ({nx},{ny});")
                nx, ny = mynet[dev["nets"][1]]["position"]
                print(f"\\draw ({dev['name']}.D) to ({nx},{ny});")
                nx, ny = mynet[dev["nets"][2]]["position"]
                print(f"\\draw ({dev['name']}.G) to ({nx},{ny});")

    # Draw text nodes (each uses a unique local bounding box for bbox extraction)
    for t in text_labels:
        rot = random.choice([-10, 0, 10])
        color = random.choice(["black", "black", "gray", "blue!60", "red!60"])
        font_size = random.choice(["\\scriptsize", "\\footnotesize", "\\small"])
        print(
            f"\\node[{color}, rotate={rot}, local bounding box={t['name']}] "
            f"at ({t['x']:.2f},{t['y']:.2f}) "
            f"{{{font_size} {t['text']}}};"
        )

    # --- Export device bboxes as YOLO labels ---
    # For each device's local bounding box (devNameBB), compute normalized YOLO format:
    # (class_id, x_center, y_center, width, height) relative to current bounding box.
    for dev in devices:
        print(
            f"\\path ({dev['name']}BB.south west);"
            + r" \pgfgetlastxy{\rOneMinX}{\rOneMinY}"
        )
        print(
            f"\\path ({dev['name']}BB.north east);"
            + r" \pgfgetlastxy{\rOneMaxX}{\rOneMaxY}"
        )
        print(
            r"""
\pgfpointanchor{current bounding box}{north west} \pgfgetlastxy{\pgNWx}{\pgNWy}
\newdimen\canvaswidth
\newdimen\canvasheight
\path (current bounding box.south west); \pgfgetlastxy{\canvasminx}{\canvasminy}
\path (current bounding box.north east); \pgfgetlastxy{\canvasmaxx}{\canvasmaxy}
\pgfmathsetlength{\canvaswidth}{\canvasmaxx-\canvasminx}
\pgfmathsetlength{\canvasheight}{\canvasmaxy-\canvasminy}
\pgfmathsetmacro{\widthratio}{(\rOneMaxX-\rOneMinX)/(\canvaswidth)}
\pgfmathsetmacro{\heightratio}{(\rOneMaxY-\rOneMinY)/(\canvasheight)}
\pgfmathsetmacro{\xpositionratio}{(\rOneMinX+\rOneMaxX-\canvasminx-\canvasminx)/2/(\canvaswidth)}
\pgfmathsetmacro{\ypositionratio}{1-(\rOneMinY+\rOneMaxY-\canvasminy-\canvasminy)/2/(\canvasheight)}
"""
        )

        # Map device type to YOLO class id
        if dev["type"] == "PMOS":
            myClass = "0"
        elif dev["type"] == "NMOS":
            myClass = "1"
        elif dev["type"] == "NPN":
            myClass = "2"
        elif dev["type"] == "PNP":
            myClass = "3"
        elif dev["type"] == "Inducer":
            myClass = "4"
        elif dev["type"] == "Diode":
            myClass = "5"
        elif dev["type"] == "Resistor":
            myClass = "6"
        elif dev["type"] == "Capacitor":
            myClass = "7"
        elif dev["type"] == "Ground":
            myClass = "8"
        elif dev["type"] == "Voltage":
            myClass = "9"
        elif dev["type"] == "Current":
            myClass = "10"
        else:
            myClass = "ERROR"

        print(
            r" \immediate\write\posfile{"
            + myClass
            + r"\space \xpositionratio \space \ypositionratio \space \widthratio \space \heightratio }"
        )

    # --- Export text bboxes as YOLO labels ---
    # Each text node was created with local bounding box = t["name"].
    for t in text_labels:
        box_name = t["name"]
        print(
            f"\\path ({box_name}.south west);"
            + r" \pgfgetlastxy{\rOneMinX}{\rOneMinY}"
        )
        print(
            f"\\path ({box_name}.north east);"
            + r" \pgfgetlastxy{\rOneMaxX}{\rOneMaxY}"
        )
        print(
            r"""
\pgfpointanchor{current bounding box}{north west} \pgfgetlastxy{\pgNWx}{\pgNWy}
\newdimen\canvaswidth
\newdimen\canvasheight
\path (current bounding box.south west); \pgfgetlastxy{\canvasminx}{\canvasminy}
\path (current bounding box.north east); \pgfgetlastxy{\canvasmaxx}{\canvasmaxy}
\pgfmathsetlength{\canvaswidth}{\canvasmaxx-\canvasminx}
\pgfmathsetlength{\canvasheight}{\canvasmaxy-\canvasminy}
\pgfmathsetmacro{\widthratio}{(\rOneMaxX-\rOneMinX)/(\canvaswidth)}
\pgfmathsetmacro{\heightratio}{(\rOneMaxY-\rOneMinY)/(\canvasheight)}
\pgfmathsetmacro{\xpositionratio}{(\rOneMinX+\rOneMaxX-\canvasminx-\canvasminx)/2/(\canvaswidth)}
\pgfmathsetmacro{\ypositionratio}{1-(\rOneMinY+\rOneMaxY-\canvasminy-\canvasminy)/2/(\canvasheight)}
"""
        )
        print(
            r" \immediate\write\posfile{"
            + TEXT_CLASS_ID
            + r"\space \xpositionratio \space \ypositionratio \space \widthratio \space \heightratio }"
        )

    # Close LaTeX environment and the YOLO label output stream
    print(
        r"""

\end{circuitikz}
\immediate\closeout\posfile
\end{document}
"""
    )
