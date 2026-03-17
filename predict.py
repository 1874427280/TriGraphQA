import argparse
import os
import torch
import numpy as np
from os.path import join
from network.net import mynet
import warnings

warnings.filterwarnings("ignore")

def predict_unknown_targets(graph_dir, output_dir, model_path):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mynet()
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"], strict=False)
    net.to(device)
    net.eval()

    summary_results = []
    pt_files = [f for f in os.listdir(graph_dir) if f.endswith('_graphs.pt')]

    with torch.no_grad():
        for pt_file in pt_files:
            file_id = pt_file.replace('_graphs.pt', '')
            pt_path = join(graph_dir, pt_file)

            try:
                graph_data = torch.load(pt_path, map_location='cpu')

                graph_A = graph_data['graph_A']
                graph_B = graph_data['graph_B']
                graph_idg = graph_data['graph_idg']

                DockQ_pred = net(graph_A, graph_B, graph_idg, device)
                dockq_pred_val = float(DockQ_pred.cpu().detach().numpy())
                print(f"[{file_id}] pDockQ: {dockq_pred_val:.4f}")

                out_npz_path = join(output_dir, f"{file_id}_predict.npz")
                np.savez_compressed(out_npz_path, DockQ_pred=np.array(dockq_pred_val, dtype=np.float16))
                summary_results.append((file_id, dockq_pred_val))

            except Exception as e:
                print(f"[{file_id}]: {e}")
                continue

    # 汇总输出
    csv_path = join(output_dir, "DockQ_predictions_summary.csv")
    with open(csv_path, 'w') as f:
        f.write("PDB_ID,Predicted_DockQ\n")
        summary_results.sort(key=lambda x: x[1], reverse=True)
        for pid, score in summary_results:
            f.write(f"{pid},{score:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DockQ scores using trained graph model.")

    parser.add_argument("--graph_dir", type=str, required=True, help="Directory containing *_graphs.pt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save prediction results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pkl model checkpoint")

    args = parser.parse_args()

    # 调用预测函数
    predict_unknown_targets(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        model_path=args.model_path
    )