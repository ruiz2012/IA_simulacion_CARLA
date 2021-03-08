import glob
import os
import sys
import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time
import math
from tensorboardX import SummaryWriter

from librerys.cnn import CNN
from librerys.decay_schedule import LinearDecaySchedule
from librerys.experience_memory import ExperienceMemory, Experience

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
#Contador global de ejecuciones
global_step_num = 0

# Habilitar entrenamiento por gráfica o CPU
use_cuda = True
device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

# Habilitar la semilla aleatoria para poder reproducir el experimento a posteriori
seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)


writer = SummaryWriter()

STEPS_PER_EPISODE = 300
SHOW_PREVIEW = False
SECONDS_PER_EPISODE=20
#MAX_NUM_EPISODES = 100
MAX_NUM_EPISODES = 200
clip_reward=True
use_target_network=True
load_trained_model=True
lista_ecu=[]
lista_error=[]
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0

    actor_list = []

    front_camera = None
    collision_hist = []

    def __init__(self):
        im_width = 84
        im_height = 84
        self.im_width = im_width
        self.im_height = im_height
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0) 
        self.world = self.client.get_world()
        blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter('model3')[0]
    
    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')


        transform = carla.Transform(carla.Location(x=5, z=2))

        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))      

        return self.front_camera       
        
    def collision_data(self, event):
        self.collision_hist.append(event)
        

    def process_img(self, image):
        i2 = np.array(image.raw_data)
        i2 = i2.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        i3 = i3.mean(2)
        i3 = i3.astype(np.float32)
        i3 *= 1.0/255.0
        i3 = np.reshape(i3, [1,84,84])
        i3 = i3[np.newaxis, ...]
        self.front_camera = i3
    def obs_shape(self):
        obs = self.front_camera.shape
        return obs
    def action_space(self):
        action_space_name = ["throttle center","throttle left","throttle right","brake"]
        return action_space_name
    
    def step(self, action):
        if action == 3:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            fre = 0

        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, brake=0.0))
            fre = 1
            
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT, brake=0.0))
            fre = 0
            
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT, brake=0.0))
            fre = 0
            
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if len(self.collision_hist) != 0:
            done = True
            reward = -2000
        elif kmh < 20:
            done = False
            reward = -10
            if kmh > 50:
                kmh = 50
        elif fre == 1:
            done = False
            reward = 500
        else:
            done = False
            reward = 500
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            reward = 2000
            done = True

        return self.front_camera, reward, done, None

class DQNAgent(object):
    def __init__(self, obs_shape, action_shape):
        self.gamma=0.75
        self.learning_rate=0.9
        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")
        self.training_steps_completed = 0
        self.action_shape=action_shape

        self.DQN=CNN

        self.Q = self.DQN(obs_shape, action_shape, device).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate)

        if use_target_network:
            self.Q_target = self.DQN(obs_shape, action_shape, device).to(device)

        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min, 
                                                 max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0

        self.memory = ExperienceMemory(capacity = int(1000000))

    def get_action(self, obs):
        return self.policy(obs)

    def epsilon_greedy_Q(self, obs):
        self.step_num += 1
        if random.random() < self.epsilon_decay(self.step_num):
            print(self.epsilon_decay(self.step_num))
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())
            print("se activo la red neuronal")
        return action

    def learn(self, obs, action, reward, next_obs, done):
        if done:
            td_target = reward + 0.0
        else:
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))

        a=self.Q(obs)
        td_error = a[:,action]-td_target
        self.Q_optimizer.zero_grad()
        td_error.backward()
        writer.add_scalar("DQL/TD_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()

    def replay_experience(self, batch_size = None):
        batch_size = batch_size if batch_size is not None else 32
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1

    def learn_from_batch_experience(self, experiences):
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)/255.0
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        

        if clip_reward == True:
            reward_batch = np.sign(reward_batch)
        next_obs_batch = np.array(batch_xp.next_obs)/255.0
        done_batch = np.array(batch_xp.done)

        for i in range(31):
            if use_target_network == True:
                if self.step_num % 2000 == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                td_target = reward_batch + ~done_batch * \
                            np.tile(self.gamma, len(next_obs_batch)) #* \
                td_target = td_target[i]*self.Q_target(next_obs_batch[i]).max(1)[0].data.cpu().numpy()
                lista_ecu.append(td_target)

            else:
                td_target = reward_batch + ~done_batch * \
                            np.tile(self.gamma, len(next_obs_batch)) #* \
                td_target = td_target[i]*self.Q_target(next_obs_batch[i]).max(1)[0].data.cpu().numpy()
                lista_ecu.append(td_target)
                
        td_target = torch.Tensor(lista_ecu)
        td_target = td_target.to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        
        for j in range(31):
            td_error = self.Q(obs_batch[j]).gather(1, action_idx[j].view(-1,1))-(td_target[j].float().unsqueeze(1))
            td_error.mean().backward()
        self.Q_optimizer.zero_grad()
        self.Q_optimizer.step()

    def save(self, env_name):
        file_name = "trained_models/"+"DQL_"+env_name+".pt"
        agent_state = {"Q": self.Q.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en : ", file_name)
        
        
    def load(self, env_name):
        file_name = "trained_models/"+"DQL_"+env_name+".pt"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Cargado del modelo Q desde", file_name,
              "que hasta el momento tiene una mejor recompensa media de: ",self.best_mean_reward,
              " y una recompensa máxima de: ", self.best_reward)
        
if __name__=='__main__':
    env_name = "AgentTrain"
    env = CarEnv()
    obs = env.reset()
    act = env.action_space()
    obs_shape = env.obs_shape()
    action_shape = len(act)
    agent = DQNAgent(obs_shape,action_shape)
    episode_rewards = list()        
    
    previous_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    
    if load_trained_model:
        try:
            agent.load(env_name)
            previous_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("ERROR: no existe ningún modelo entrenado para este entorno. Empezamos desde cero")

    episode = 0
    
    while global_step_num < MAX_NUM_EPISODES:
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        env.collision_hist = []
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            total_reward += reward
            step += 1
            
                    
            if done is True:
                
                episode += 1
                episode_rewards.append(total_reward)
                global_step_num += 1

                if total_reward > agent.best_reward:
                    agent.best_reward = total_reward
                for actor in env.actor_list:
                    actor.destroy()
            if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew: 
                    num_improved_episodes_before_checkpoint += 1

            if num_improved_episodes_before_checkpoint >= 100:
                    previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_name)
                    num_improved_episodes_before_checkpoint = 0

            print("\n Episodio #{} finalizado con {} iteraciones. recompensa = {}, recompensa media = {:.2f}, mejor recompensa = {}".
            format(episode, step+1, total_reward, np.mean(episode_rewards), agent.best_reward))

            writer.add_scalar("main/ep_reward", total_reward, global_step_num)
            writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
            writer.add_scalar("main/max_ep_reward", agent.best_reward, global_step_num)

            if agent.memory.get_size() >= 5*100000:
                agent.replay_experience()

            #break
