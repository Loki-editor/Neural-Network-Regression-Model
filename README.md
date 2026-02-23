# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="574" height="374" alt="image" src="https://github.com/user-attachments/assets/236974c6-d7b5-426b-bd93-cbcf21fc20ed" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:  LOKESH S

### Register Number: 212224240079

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss= criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```

### Dataset Information
<img width="124" height="395" alt="image" src="https://github.com/user-attachments/assets/0c84252b-ee12-41b7-b721-a4c474d2eb03" />


### OUTPUT
<img width="351" height="181" alt="image" src="https://github.com/user-attachments/assets/61ea5407-0925-469e-a25f-7fba1164ffc9" />


### Training Loss Vs Iteration Plot
<img width="606" height="456" alt="image" src="https://github.com/user-attachments/assets/41f572a5-c01f-4f14-893d-2f5981ec40bd" />


### New Sample Data Prediction
<img width="768" height="107" alt="image" src="https://github.com/user-attachments/assets/552f5cac-1a11-4251-81a1-6e83d882bb3b" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
