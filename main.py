from utils import *
from models import *
import argparse
# main.py

import argparse

def main():
    parser = argparse.ArgumentParser(description='kvasir Detection')

    # train subcommand
    parser_cmd = parser.add_subparsers(dest='command', help='Available commands')
    train_cmd = parser_cmd.add_parser('train', help='Train the model')  # 创建一个命令对象
    train_cmd.add_argument('--model', type=str, default='AlexNet', help='Type of the model used')
    train_cmd.add_argument('--data', type=str, default='data/', help='Path to the processed data directory')
    train_cmd.add_argument('--dataset', type=str, default='data/kvasir-dataset-v2', help='Path to the dataset directory')
    train_cmd.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    train_cmd.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    train_cmd.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
    train_cmd.add_argument('--pretrained', type=bool, default=False, help='Use pretrained parameters')
    train_cmd.add_argument('--num_classes', type=int, default=8, help='Number of the classes')

    # predict subcommand
    predict_cmd = parser_cmd.add_parser('predict', help='Predict using the trained model')
    predict_cmd.add_argument('--img', type=str, default="data\ex.jpg", help='Path to the image for prediction')
    predict_cmd.add_argument('--num_classes', type=int, default=8, help='Number of the classes')
    predict_cmd.add_argument('--model', type=str, default='AlexNet', help='Type of the model used')
    predict_cmd.add_argument('--dataset', type=str, default='data/kvasir-dataset-v2', help='Path to the dataset directory')
    
    
    gui_cmd = parser_cmd.add_parser('gui', help='A Gui for Detection')
    
    
    
    args = parser.parse_args()
    if not args.command:
        print("Error: Please provide a valid command. Use '--help' for more information.")
        parser.print_help()
    elif args.command == 'gui':
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    else:
        # Initialize and train the ResNet model
        if args.model == 'AlexNet':
            net = AlexNet(num_classes=args.num_classses)
        elif args.model == 'DenseNet':
            net = DenseNet(num_classes=args.num_classes)
        else:
            print("ERROR: No Such Model")
            return
        if args.command == 'predict':
            label = predict(model=net, dataset=args.dataset, image_path=args.img)
        elif args.command == 'train':
            trainset, validset, testset = get_loaders(img_size=net.input_size, batch_size=args.batch_size)
            train(net, trainset, validset, testset, epochs=args.epochs, pretrained=args.pretrained, initial_lr=args.learning_rate)

if __name__ == '__main__':
    main()

