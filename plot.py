import matplotlib.pyplot as plt
datasets = ['minst-d', 'minst-f', 'cifar-10', 'cifar-100-c', 'cifar-100-f']
accuracy = [93.43, 76.59, 0, 0, 0]

# with open("cnn_data.txt") as file:
#     content = file.readlines()
#     for line in content:
#         if len(line.rstrip()) != 0:
#             accuracy.append(float(line.rstrip()))

print(accuracy)
plt.bar(datasets, accuracy)
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.title("ANN classification accuracy on different datasets")
plt.show()
