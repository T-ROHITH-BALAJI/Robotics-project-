import pybullet as p
import pybullet_data
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ========== CONFIGURATIONS ==========
WORKSPACE = {
    'master': (0.5, 0, 0),
    'fruits': (0, -0.6, 0),
    'vegetables': (0, 0.6, 0)
}
PLATE_SIZE = 0.4  # 40cm x 40cm plates
BLOCK_SIZE = 0.05
APPROACH_HEIGHT = 0.2
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
HOME_JOINT_POSITIONS = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
MOVE_SPEED = 0.005  # smaller = slower, smoother
CONTROL_FORCE = 100  # lower forces for smooth moves

model = tf.keras.models.load_model('fruit_veg_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('max_length.pickle', 'rb') as f:
    MAX_SEQ_LENGTH = pickle.load(f)

class PickAndPlace:
    def __init__(self):
        # Connect and set up simulation
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        self.load_environment()
        self.load_robot()
        self.spawn_blocks()
    
    def load_environment(self):
        p.loadURDF("plane.urdf")
        
        plate_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[PLATE_SIZE/2, PLATE_SIZE/2, 0.02])
        plate_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[PLATE_SIZE/2, PLATE_SIZE/2, 0.02], rgbaColor=[0.8, 0.8, 0.8, 1])
        
        for pos in WORKSPACE.values():
            p.createMultiBody(0, plate_shape, plate_visual, basePosition=pos)
    
    def load_robot(self):
        flags = p.URDF_USE_SELF_COLLISION
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=flags)
        self.ee_link = 11  # Panda's end-effector link index
        self.reset_to_home()
        self.control_gripper(open=True)
    
    def reset_to_home(self):
        for i in range(7):
            p.resetJointState(self.robot, i, HOME_JOINT_POSITIONS[i])
    
    def spawn_blocks(self):
        self.blocks = []
        block_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BLOCK_SIZE/2]*3)
        
        colors = {'red': [1, 0, 0, 1], 'green': [0, 1, 0, 1]}
        offsets = np.linspace(-0.15, 0.15, 5)
        
        for i, x_offset in enumerate(offsets):
            # Red blocks (fruits)
            pos_red = [WORKSPACE['master'][0] + x_offset, WORKSPACE['master'][1] - 0.1, BLOCK_SIZE/2]
            block_red = p.createMultiBody(0.2, block_shape, -1, pos_red)
            p.changeVisualShape(block_red, -1, rgbaColor=colors['red'])
            self.blocks.append({'id': block_red, 'color': 'red', 'picked': False})
            
            # Green blocks (vegetables)
            pos_green = [WORKSPACE['master'][0] + x_offset, WORKSPACE['master'][1] + 0.1, BLOCK_SIZE/2]
            block_green = p.createMultiBody(0.2, block_shape, -1, pos_green)
            p.changeVisualShape(block_green, -1, rgbaColor=colors['green'])
            self.blocks.append({'id': block_green, 'color': 'green', 'picked': False})
    
    def move_to(self, target_pos, steps=240):
        # Maintain constant orientation: downward-facing
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        start_pos = p.getLinkState(self.robot, self.ee_link)[0]

        # Bézier control points for smooth arc
        control1 = [start_pos[0], start_pos[1], start_pos[2] + 0.15]
        control2 = [target_pos[0], target_pos[1], target_pos[2] + 0.15]

        def bezier(t, p0, p1, p2, p3):
            return [(1 - t)**3 * p0[i] + 3 * (1 - t)**2 * t * p1[i] + 3 * (1 - t) * t**2 * p2[i] + t**3 * p3[i] for i in range(3)]

        # ---- Print optimization and IK plan ----
        print(f"\n[INFO] Optimizing the trajectory based on shortest path.")
        print(f"[INFO] Target: {np.round(target_pos, 3)}")

        ik_solution = p.calculateInverseKinematics(self.robot, self.ee_link, target_pos, orn, maxNumIterations=200, residualThreshold=1e-5)
        ik_joints = np.round(ik_solution[:7], 4)
        print(f"[INFO] Following Inverse Kinematics Metrics: {ik_joints}")

        # ---- Smooth motion following Bézier ----
        for i in range(steps):
            t = (i + 1) / steps
            interp = bezier(t, start_pos, control1, control2, target_pos)

            joints = p.calculateInverseKinematics(self.robot, self.ee_link, interp, orn, maxNumIterations=200, residualThreshold=1e-5)
            for j in range(7):
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                        targetPosition=joints[j], force=CONTROL_FORCE)

            p.stepSimulation()
            time.sleep(MOVE_SPEED)
    
    def control_gripper(self, open=True):
        target = GRIPPER_OPEN if open else GRIPPER_CLOSED
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, targetPosition=target, force=20)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, targetPosition=target, force=20)
        
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1./240.)
    
    def pick_block(self, block):
        block_pos, _ = p.getBasePositionAndOrientation(block['id'])
        approach = [block_pos[0], block_pos[1], block_pos[2] + APPROACH_HEIGHT]
        grasp = [block_pos[0], block_pos[1], block_pos[2] + 0.01]
        
        self.move_to(approach)
        self.move_to(grasp)
        
        self.control_gripper(open=False)
        
        contacts = p.getContactPoints(self.robot, block['id'])
        if contacts:
            self.grasp_constraint = p.createConstraint(
                self.robot, self.ee_link,
                block['id'], -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            )
        
        self.move_to(approach)
    
    def place_block(self, plate_name):
        target_pos = list(WORKSPACE[plate_name])
        target_pos[2] = BLOCK_SIZE/2
        
        approach = [target_pos[0], target_pos[1], target_pos[2] + APPROACH_HEIGHT]
        drop = [target_pos[0], target_pos[1], target_pos[2] + 0.01]
        
        self.move_to(approach)
        self.move_to(drop)
        
        if hasattr(self, 'grasp_constraint'):
            p.removeConstraint(self.grasp_constraint)
            del self.grasp_constraint
        
        self.control_gripper(open=True)
        self.move_to(approach)
        self.reset_to_home_smooth()
    
    def reset_to_home_smooth(self):
        current_pos = p.getLinkState(self.robot, self.ee_link)[0]
        home_pos = [0, 0, 0.6]  # float to the center
        
        self.move_to(home_pos)
        
        for i in range(7):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL,
                                    targetPosition=HOME_JOINT_POSITIONS[i], force=CONTROL_FORCE)
        for _ in range(240):
            p.stepSimulation()
            time.sleep(MOVE_SPEED)
    
    def get_nearest_block(self, color):
        ee_pos = p.getLinkState(self.robot, self.ee_link)[0]
        nearest = None
        min_dist = float('inf')
        
        for block in self.blocks:
            if not block['picked'] and block['color'] == color:
                block_pos, _ = p.getBasePositionAndOrientation(block['id'])
                dist = np.linalg.norm(np.array(ee_pos) - np.array(block_pos))
                if dist < min_dist:
                    min_dist = dist
                    nearest = block
        return nearest

    def classify_word(self, word):
        """Classify input word using CNN model"""
        sequence = tokenizer.texts_to_sequences([word.lower()])
        padded = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH, padding='post')
        prediction = model.predict(padded, verbose=0)
        return 'vegetable' if prediction[0][0] > 0.5 else 'fruit'

    def run(self):
        try:
            while True:
                word = input("\nEnter a fruit/vegetable name: ").strip()
                
                # Classify using CNN
                category = self.classify_word(word)
                print(f"Classification: {category}")
                
                color = 'red' if category == 'fruit' else 'green'
                plate = 'fruits' if category == 'fruit' else 'vegetables'

                block = self.get_nearest_block(color)
                if block:
                    print(f"Picking {color} block...")
                    self.pick_block(block)
                    print(f"Placing on {plate} plate...")
                    self.place_block(plate)
                    block['picked'] = True
                    print("Done!\n")
                else:
                    print(f"No more {color} blocks left!")

        except KeyboardInterrupt:
            p.disconnect()

if __name__ == "__main__":
    simulation = PickAndPlace()
    simulation.run()
