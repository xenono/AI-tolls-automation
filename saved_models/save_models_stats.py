import os, json

file = open("models-statistics.json", "r+")
models = json.load(file)

for model_name in os.listdir("./"):
    if not os.path.isdir("./" + model_name): continue
    if model_name == "VGG16": continue
    models_epochs = []
    for epoch in os.listdir("./" + model_name):
        epoch_data = epoch[11:-5].split("-")
        epoch_number = int(epoch_data[0][1:])
        epoch_loss = float(epoch_data[1][1:])
        epoch_accuracy = float(epoch_data[2][1:])
        epoch_val_loss = float(epoch_data[3][2:])
        epoch_val_accuacy = float(epoch_data[4][2:])

        epoch_json_data = {
            "epoch": epoch_number,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_accuacy
        }

        models_epochs.append(epoch_json_data)
    models_epochs.sort(key= lambda f: f["epoch"])
    models[model_name] = models_epochs

file.seek(0)
json.dump(models, file, indent=4)
file.truncate()
file.close()


