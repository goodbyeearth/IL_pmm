from pommerman.agents import BaseAgent
import action_prune
from featurize import feat_v2
from model import FCConv5
import torch


class ImitateAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(ImitateAgent, self).__init__(*args, **kwargs)
        save_path = '../158save/FCConv_epoch26_of_30.th'

        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
        self.model = FCConv5()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.prev = (None, None)

    def act(self, obs, action_space):
        board_obs, flat_obs = feat_v2(obs)
        feat = {'board': torch.Tensor(board_obs).unsqueeze(0),
                'flat': torch.Tensor(flat_obs).unsqueeze(0)}
        act = self.model(feat)
        act = self.modify_act(obs=obs, act=act, prev=self.prev)

        self.set_prev2obs(prev=self.prev, obs=obs)

        return int(act), 2, 2

    def modify_act(self, obs, act, prev):
        valid_actions = action_prune.get_filtered_actions(obs.copy(), prev_two_obs=prev)
        for i in range(6):
            if i not in valid_actions:
                act[0][i] = -10000

        final_act = act.argmax().item()
        return final_act

    def set_prev2obs(self, prev, obs):
        import copy
        old_old, old = prev
        old_old = copy.deepcopy(old)
        old = copy.deepcopy(obs)
        self.prev = (old_old, old)
