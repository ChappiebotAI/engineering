import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

gold_actions = {
    'A': ['next', 'next', 'end'],
    'B': ['next', 'end'],
    'C': ['next', 'next', 'next', 'end'],
    'D': ['next', 'next', 'end'],
    'E': ['next', 'next', 'next', 'next', 'end'],
    'F': ['next', 'next', 'next', 'end'],
    'G': ['end']
}

lexicons = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "next",
    "end"
]

state_actions = ["next", "end"]

MAX_LEN_SEQ = 8


class LSTM_REINFORCE(nn.Module):
    def __init__(self, using_cuda):
        super(LSTM_REINFORCE, self).__init__()
        self.embed_size = 10
        self.n_hidden= 30
        self.vocab_size = len(lexicons)
        self.n_action = 2
        self.discount_factor = 1.

        self.lexicons_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.action_lstm_cell = nn.LSTMCell(self.embed_size, self.n_hidden)
        self.hidden2action = nn.Linear(self.n_hidden, self.n_action)
        self.value_regression = nn.Linear(self.n_hidden, 1)

        self.using_cuda = using_cuda

    def lstm_init_hidden(
            self,
            batch_size=1,
            n_unit=20):
        hidden = [autograd.Variable(torch.zeros(batch_size, n_unit)),
                  autograd.Variable(torch.zeros(batch_size, n_unit))]  # Init hidden + state
        if self.using_cuda: hidden[0] = hidden[0].cuda(); hidden[1] = hidden[1].cuda();
        return hidden

    def _sampling_action(self, s, ihc):
        s = torch.LongTensor([s])
        s = autograd.Variable(s)
        if self.using_cuda: s = s.cuda()
        w_embed = self.lexicons_embed(s)
        h, c = self.action_lstm_cell(w_embed, ihc)
        value = self.value_regression(h)
        prob_logs = self.hidden2action(h)
        action_probs = F.softmax(prob_logs)

        action = torch.multinomial(action_probs, 1).cpu().data.numpy()[0][0]

        return value, action, torch.log(action_probs[0][action]), h, c

    def _max_action(self, s, ihc):
        s = torch.LongTensor([s])
        s = autograd.Variable(s)
        if self.using_cuda: s = s.cuda()
        w_embed = self.lexicons_embed(s)
        h, c = self.action_lstm_cell(w_embed, ihc)
        value = self.value_regression(h)
        prob_logs = self.hidden2action(h)
        action_probs = F.softmax(prob_logs)
        max_value, idx = action_probs.max(1)

        action = idx[0].cpu().data.numpy()[0]
        return value, action, h, c

    def sampling_episode(self, batch):
        episodes = []

        for s in batch:
            episode = []
            state = [lexicons[s],]
            ga = gold_actions[state[0]]

            ihc = self.lstm_init_hidden(n_unit=self.n_hidden)
            value, action, prob_log, h, c = self._sampling_action(s, ihc)  # sampling init action
            na = state_actions[action]
            next_state = state + [na,]  # compute next state
            i = 0
            g_t = 0.

            while next_state[-1] != "end" and i < MAX_LEN_SEQ:
                if i < len(ga) and next_state[-1] == ga[i]:
                    reward = 0.5
                else:
                    reward = -1.

                g_t = g_t * self.discount_factor + reward
                step = (state, na, g_t, value, prob_log)
                episode.append(step) # Store a step
                state = next_state

                value, action, prob_log, h, c = self._sampling_action(lexicons.index(state[-1]), (h, c))
                i += 1
                na = state_actions[action]
                next_state = state + [na, ]  # compute next state

            if i == (len(ga) - 1) and next_state[-1] == ga[i]:
                reward = 2.
            elif next_state[-1] == "end":
                reward = -0.5
            else:
                reward = -2.
            g_t = g_t * self.discount_factor + reward
            step = (state, na, g_t, value, prob_log)
            episode.append(step)
            episodes.append(episode)
            continue

        return episodes

    def calc_loss(self, episodes):
        loss = 0.
        num_step = 0
        for ep in episodes:
            num_step += len(ep)
            for i,step in enumerate(ep):
                # print(step)
                #loss -= (step[2] * step[4] * math.pow(self.discount_factor, i))   # uncommen if dont use baseline
                loss -= ((step[2] - step[3]) * step[4] * math.pow(self.discount_factor, i) - (step[2] - step[3]) * (step[2] - step[3]))
        loss /= num_step

        return loss

    def forward(self, state):
        print("State: %s" % state)
        istate = lexicons.index(state)
        ihc = self.lstm_init_hidden(n_unit=self.n_hidden)
        value, action, h, c = self._max_action(istate, ihc)
        na = state_actions[action]
        ina = lexicons.index(na)
        print("Action: %s" % na, "G_t: %0.3f" % value)
        i = 0
        while na != "end" and i < 20:
            i += 1
            value, action, h, c = self._max_action(ina, (h, c))
            na = state_actions[action]
            ina = lexicons.index(na)
            print("Action: %s" % na, "G_t: %0.3f" % value)


def convert_batch(raw):
    batch = []
    for r in raw:
        batch.append(lexicons.index(r))

    return batch


def print_eposides(eposides):
    for ep in eposides:
        print("\n")
        for step in ep:
            print(step[:3])
            pass
    pass

if __name__ == "__main__":
    using_cuda = False
    lr = 0.01
    lstm_reinfore = LSTM_REINFORCE(using_cuda=using_cuda)
    raw = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    batch = convert_batch(raw)

    if using_cuda: lstm_reinfore.cuda()

    optimizer = optim.SGD(lstm_reinfore.parameters(),
                          lr=lr,
                          momentum=0.9)
    for i in range(1000):
        episodes = lstm_reinfore.sampling_episode(batch)
        #print_eposides(episodes)

        loss = lstm_reinfore.calc_loss(episodes)

        if using_cuda:
            if not isinstance(loss, float): loss = loss.cuda()

        lstm_reinfore.zero_grad()
        if not isinstance(loss, float):
            loss.backward()
            optimizer.step()

        print("Step:%d - Loss:%0.3f" % (i, loss))

    # Test
    lstm_reinfore('A')
    print("\n")
    lstm_reinfore('B')
    print("\n")
    lstm_reinfore('C')
    print("\n")
    lstm_reinfore('D')
    print("\n")
    lstm_reinfore('E')
    print("\n")
    lstm_reinfore('F')
    print("\n")
    lstm_reinfore('G')
