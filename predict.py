from load_model import load_checkpoint
from process_input import process_image
from label_mapping import get_label_mapping
import torch.nn.functional as F
import torch

import argparse

def predict(image_path, checkpoint_path, category_names, topk=5, gpu=False):

    if gpu and not torch.cuda.is_available():
        print("GPU is not available. Using CPU for prediction.")
        gpu = False

    model, _, _ = load_checkpoint(checkpoint_path, gpu)

    if gpu:
        model.cuda()

    img_tensor = torch.tensor(process_image(image_path))
    img_tensor = img_tensor.unsqueeze(0) # add batch dim
    if gpu:
        img_tensor = img_tensor.cuda()
    
    model.eval()
        
    with torch.no_grad():
        outputs = model(img_tensor)
        ps = F.softmax(outputs, dim=1)

        top_p, top_class = ps.topk(topk, dim=1)
    
    top_p = top_p.squeeze().tolist()
    top_class = top_class.squeeze().tolist()

    top_class = [c + 1 for c in top_class]

    predicted_class = ""
    max_val = -1

    for prob, cls in zip(top_p, top_class):
        if prob > max_val:
            max_val = prob
            predicted_class = cls

    print("Top classes are =>", top_class)
    print("Top probabilities are =>", top_p)

    print("Predicted class is =>", get_label_mapping(category_names)[predicted_class])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image using a pre-trained model.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top classes to return.")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to the category names file.")
    parser.add_argument("--gpu", action="store_true", help="Flag to use GPU for prediction.")

    args = parser.parse_args()
    
    predict(
        image_path=args.image_path, 
        checkpoint_path=args.checkpoint_path,
        topk=args.topk,
        gpu=args.gpu,
        category_names=args.category_names 
    )

    


