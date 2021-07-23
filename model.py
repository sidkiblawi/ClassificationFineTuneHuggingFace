import torch
import transformers


class HuggingFaceCustomClassifier(torch.nn.Module):

    def __init__(self, hugging_face_encoder, hugging_face_config, num_classes):
        super(HuggingFaceCustomClassifier, self).__init__()

        self.hugging_face_encoder = hugging_face_encoder

        # dropout layer
        self.dropout = torch.nn.Dropout(0.1)

        # relu activation function
        self.relu = torch.nn.ReLU()

        # dense layer 1
        self.fc1 = torch.nn.Linear(hugging_face_config.hidden_size, 512)

        # dense layer 2 (Output layer)
        self.fc2 = torch.nn.Linear(512, num_classes)

        # softmax activation function
        self.softmax = torch.nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.hugging_face_encoder(sent_id, attention_mask=mask).pooler_output
        print(cls_hs.shape)
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x