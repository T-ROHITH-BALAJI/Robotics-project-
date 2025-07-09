# Robotics-project-
CNN-Based Fruit and Vegetable Sorting with Robotic Arm in PyBullet

CNN-based Classification: A trained CNN model is used to classify fruits and vegetables based on user input (text)
PyBullet Simulation: A robotic arm with a gripper is simulated in PyBullet to execute pick-and-place operations. The simulation includes dynamic modeling of the robotic arm, gripper mechanics, and scene interaction.
Autonomous Sorting: The system autonomously identifies and sorts objects (represented by colored blocks: red for fruits, green for vegetables) into designated trays.
User Interaction: The system allows users to input the name of a fruit or vegetable, triggering the sorting task in the simulation.




Technologies Used :

->TensorFlow/Keras: For building and loading the CNN classification model.
->PyBullet: For robot and object simulation with physics.
->URDF: For 3D modeling of the robot and environment elements.
->Python/Numpy: Primary scripting language for logic and control, with Numpy used for numerical operations.

Project Structure
cnn_model.ipynb: Jupyter Notebook for CNN model development and training.
fruit_veg_model.h5: Trained Keras/TensorFlow model for fruit and vegetable classification.
max_length.pickle: Pickled file containing the maximum sequence length used for CNN input padding.
requirements.txt: Lists Python dependencies required to run the project.
robot_control.py: The main Python script that integrates the CNN model with the PyBullet robotic arm simulation for sorting.
tokenizer.pickle: Pickled file for the Keras Tokenizer used in character-to-index mapping for CNN input.
templates/: (Potentially) Contains web templates if this project has a web interface
__pycache__/: Python cache directory.

How to run :
run this ...!
python robot_control.py
