import os
import numpy as np
import networkx as nx
from pathlib import Path
from ultralytics import YOLO
import cv2


class CircuitGraphBuilder:
    
    def __init__(self):
        self.class_names = {
            0: 'PMOS', 1: 'NMOS', 2: 'NPN', 3: 'PNP',
            4: 'Inducer', 5: 'Diode', 6: 'Resistor', 7: 'Capacitor',
            8: 'Ground', 9: 'Voltage', 10: 'Current', 11: 'Text_Label'
        }
        
    def parse_cir_file(self, cir_path):
    #  解析netlist .cir文件
        G = nx.Graph()
        
        with open(cir_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
                
            component_name = parts[0]
            component_type = component_name[0].upper()
            
            # 元器件种类
            type_map = {
                'M': 'PMOS' if 'P' in component_name else 'NMOS',
                'Q': 'NPN',
                'R': 'Resistor',
                'C': 'Capacitor',
                'L': 'Inducer',
                'D': 'Diode',
                'V': 'Voltage',
                'I': 'Current'
            }
            
            comp_type = type_map.get(component_type, 'Unknown')
            

            G.add_node(component_name, type=comp_type)
            

            if component_type in ['M', 'Q']:
                if len(parts) >= 4:
                    nets = parts[1:4]
            elif component_type in ['R', 'C', 'L', 'D', 'V', 'I']:
                if len(parts) >= 3:
                    nets = parts[1:3]
            else:
                continue

            for net in nets:
                if not G.has_node(net):
                    G.add_node(net, type='net')
                G.add_edge(component_name, net)
        
        return G
    
    def build_graph_from_detections(self, detections, img_width, img_height):
        # YOLO 检测边缘连接
        G = nx.Graph()
        
        components = []
        text_labels = []
        
        # 分割元器件和文本标签
        for det in detections:
            x_center, y_center, width, height, conf, cls = det
            cls = int(cls)
            
            if cls == 11:  # 文本标签bbox
                text_labels.append({
                    'x': x_center * img_width,
                    'y': y_center * img_height,
                    'w': width * img_width,
                    'h': height * img_height,
                    'conf': conf,
                    'cls': cls
                })
            else:  # 元器件
                comp_name = f"{self.class_names[cls]}_{len(components)}"
                components.append({
                    'name': comp_name,
                    'type': self.class_names[cls],
                    'x': x_center * img_width,
                    'y': y_center * img_height,
                    'w': width * img_width,
                    'h': height * img_height,
                    'conf': conf,
                    'cls': cls
                })
                G.add_node(comp_name, type=self.class_names[cls])
        
        # 进行结点连接
        proximity_threshold = 100 
        
        net_id = 0
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                dist = np.sqrt((comp1['x'] - comp2['x'])**2 + 
                              (comp1['y'] - comp2['y'])**2)
                
                if dist < proximity_threshold:
                    net_name = f"net_{net_id}"
                    if not G.has_node(net_name):
                        G.add_node(net_name, type='net')
                        net_id += 1
                    
                    G.add_edge(comp1['name'], net_name)
                    G.add_edge(comp2['name'], net_name)
        
        return G, components, text_labels


class GEDEvaluator:
    # GED 评估
    
    def __init__(self):
        self.graph_builder = CircuitGraphBuilder()
    
    def compute_ged(self, G1, G2, mode='full'):
       
        
        def node_match(n1, n2):
            if mode == 'interconnection':
                return True 
            return n1.get('type') == n2.get('type')
        
        def edge_match(e1, e2):
            if mode == 'component':
                return True  
            return True
        
        try:  # 计算GED
            ged = nx.graph_edit_distance(
                G1, G2,
                node_match=node_match if mode != 'interconnection' else None,
                edge_match=edge_match if mode != 'component' else None,
                timeout=10  
            )
            return ged
        except Exception as e:
            print(f"GED computation failed: {e}")
            return float('inf')
    
    def compute_ged_accuracy(self, ged, max_ged=100):
        # Convert GED to accuracy score (0-1)
        # Accuracy = 1 - (GED / max_possible_GED)
        # Lower GED = Higher accuracy
        accuracy = max(0, 1 - (ged / max_ged))
        return accuracy
    
    def evaluate_dataset(self, model_path, dataset_path, mode='full'):
        # 评估训练好的YOLO模型
        model = YOLO(model_path)
        
        dataset_path = Path(dataset_path)
        
        sample_folders = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
        
        if not sample_folders:
            print("No numbered folders found, trying flat structure...")
            sample_folders = [dataset_path]
        
        results = {
            'ged_scores': [],
            'accuracies': [],
            'component_counts': [],
            'edge_counts': []
        }
        
        for folder in sorted(sample_folders, key=lambda x: int(x.name) if x.name.isdigit() else 0):
            png_files = list(folder.glob('*.png'))
            if not png_files:
                continue
            
            img_path = png_files[0]  
            folder_name = folder.name
            
            cir_path = folder / f"{folder_name}.cir"
            
            if not cir_path.exists():
                print(f"Warning: {cir_path} not found, skipping {folder.name}")
                continue
            
            gt_graph = self.graph_builder.parse_cir_file(cir_path)
            
            # 进行YOLO detect
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2]
            
            yolo_results = model(img, verbose=False)[0]
            
            detections = []
            if len(yolo_results.boxes) > 0:
                boxes = yolo_results.boxes.xywhn.cpu().numpy()  
                confs = yolo_results.boxes.conf.cpu().numpy()
                classes = yolo_results.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    detections.append([*box, conf, cls])
            
            pred_graph, components, text_labels = self.graph_builder.build_graph_from_detections(
                detections, img_width, img_height
            )
            
            # 计算GED
            ged = self.compute_ged(gt_graph, pred_graph, mode=mode)
            accuracy = self.compute_ged_accuracy(ged, max_ged=100)
            
            results['ged_scores'].append(ged)
            results['accuracies'].append(accuracy)
            results['component_counts'].append(len([n for n, d in pred_graph.nodes(data=True) if d.get('type') != 'net']))
            results['edge_counts'].append(pred_graph.number_of_edges())
            
            print(f"{folder.name}: GED={ged:.2f}, Accuracy={accuracy:.4f}")
        
        # 计算各种metrics
        if results['ged_scores']:
            print("\n" + "="*60)
            print(f"Evaluation Mode: {mode.upper()}")
            print("="*60)
            print(f"Average GED: {np.mean(results['ged_scores']):.2f}")
            print(f"Average Accuracy: {np.mean(results['accuracies']):.4f}")
            print(f"Std Dev GED: {np.std(results['ged_scores']):.2f}")
            print(f"Min GED: {np.min(results['ged_scores']):.2f}")
            print(f"Max GED: {np.max(results['ged_scores']):.2f}")
            print(f"\nAverage Components Detected: {np.mean(results['component_counts']):.1f}")
            print(f"Average Edges: {np.mean(results['edge_counts']):.1f}")
        
        return results


def evaluate_component_only(model_path, dataset_path):
    model = YOLO(model_path)
    dataset_path = Path(dataset_path)
    
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    
    class_names = {
        0: 'PMOS', 1: 'NMOS', 2: 'NPN', 3: 'PNP',
        4: 'Inducer', 5: 'Diode', 6: 'Resistor', 7: 'Capacitor',
        8: 'Ground', 9: 'Voltage', 10: 'Current'
    }
    
    sample_folders = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not sample_folders:
        print("No numbered folders found, trying flat structure...")
        sample_folders = [dataset_path]
    
    for folder in sorted(sample_folders, key=lambda x: int(x.name) if x.name.isdigit() else 0):
        png_files = list(folder.glob('*.png'))
        if not png_files:
            continue
        
        img_path = png_files[0]
        folder_name = folder.name
        
        label_path = folder / f"Pagenumber{folder_name}.txt"
        
        if not label_path.exists():
            print(f"Warning: {label_path} not found, skipping {folder.name}")
            continue
        
        gt_components = {}
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls < 11: 
                        gt_components[cls] = gt_components.get(cls, 0) + 1
        
        results = model(str(img_path), verbose=False)[0]
        pred_components = {}
        
        if len(results.boxes) > 0:
            classes = results.boxes.cls.cpu().numpy()
            for cls in classes:
                cls = int(cls)
                if cls < 11:
                    pred_components[cls] = pred_components.get(cls, 0) + 1
        
        all_classes = set(gt_components.keys()) | set(pred_components.keys())
        for cls in all_classes:
            gt_count = gt_components.get(cls, 0)
            pred_count = pred_components.get(cls, 0)
            
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("="*60)
    print("Component Detection Evaluation")
    print("="*60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def main():
    """
    
    Usage:
        python ged_evaluation.py --mode full
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO circuit detection model')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                        help='Path to trained YOLO model')
    parser.add_argument('--dataset', type=str, default='dataset/val',
                        help='Path to validation dataset')
    parser.add_argument('--mode', type=str, default='component',
                        choices=['component', 'interconnection', 'full', 'all'],
                        help='Evaluation mode')
    
    args = parser.parse_args()
    
    model_path = args.model
    dataset_path = args.dataset
    mode = args.mode
    
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Mode: {mode}\n")
    
    if mode == 'component' or mode == 'all':
        print("\n" + "="*60)
        print("MODE 1: COMPONENT DETECTION ACCURACY")
        print("(GED with node weights only)")
        print("="*60)
        results_component = evaluate_component_only(model_path, dataset_path)
    
    if mode == 'interconnection' or mode == 'all':
        print("\n" + "="*60)
        print("MODE 2: INTERCONNECTION DETECTION ACCURACY")
        print("(GED with edge weights only)")
        print("="*60)
        print("WARNING: This requires .cir files for ground truth connections")
        evaluator = GEDEvaluator()
        
        test_folders = [d for d in Path(dataset_path).iterdir() if d.is_dir() and d.name.isdigit()]
        if test_folders:
            test_cir = test_folders[0] / f"{test_folders[0].name}.cir"
        else:
            test_cir = Path(dataset_path) / "1.cir"
        
        if test_cir.exists():
            results_interconnection = evaluator.evaluate_dataset(
                model_path, dataset_path, mode='interconnection'
            )
        else:
            print("ERROR: .cir files not found in dataset!")
            print("Interconnection evaluation requires netlist files.")
    
    if mode == 'full' or mode == 'all':
        print("\n" + "="*60)
        print("MODE 3: FULL NETLIST ACCURACY")
        print("(GED with equal weights for nodes and edges)")
        print("="*60)
        print("WARNING: This requires .cir files for ground truth connections")
        evaluator = GEDEvaluator()
        
        test_folders = [d for d in Path(dataset_path).iterdir() if d.is_dir() and d.name.isdigit()]
        if test_folders:
            test_cir = test_folders[0] / f"{test_folders[0].name}.cir"
        else:
            test_cir = Path(dataset_path) / "1.cir"
        
        if test_cir.exists():
            results_full = evaluator.evaluate_dataset(
                model_path, dataset_path, mode='full'
            )
        else:
            print("ERROR: .cir files not found in dataset!")
            print("Full netlist evaluation requires netlist files.")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
