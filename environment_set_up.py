import gym

class EnvironmentSetup:
    def __init__(self, env_name="MontezumaRevenge-v4"):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
