import gym

class EnvironmentSetup:
    def __init__(self, env_name="MontezumaRevenge-v4", render_mode="human"):
        self.env_name = env_name
        self.env = gym.make(self.env_name, render_mode=render_mode, frameskip=4)  # Set render_mode here
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()  # Should now render correctly
    
    def close(self):
        self.env.close()
