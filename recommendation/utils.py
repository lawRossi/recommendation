import numpy as np


class LogUniformSampler():
    def __init__(self, ntokens, padding=True):

        self.N = ntokens
        self.prob = np.zeros(ntokens)
        self.padding = padding
        self.log_N = np.log(self.N)
        self.generate_distribution()

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = {sample: i for i, sample in enumerate(samples)}

        result = list()
        for idx, label in enumerate(labels):
            if label in sample_dict:
                result.append((idx, sample_dict[label]))

        return result

    def sample(self, size, labels=None):
        x = np.random.uniform(low=0.0, high=1.0, size=size)
        samples = np.floor(np.exp(x * self.log_N)).astype(int) - 1
        if self.padding:
            samples = samples + 1

        # true_freq = self.expected_count(size, labels.tolist())
        # sample_freq = self.expected_count(size, samples)

        # return samples, true_freq, sample_freq
        return samples


if __name__ == "__main__":
    sampler = LogUniformSampler(50)
    print(sampler.sample(3))