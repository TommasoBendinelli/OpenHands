import numpy as np
from sklearn.datasets import fetch_20newsgroups


def main():
    def data_generator(subsample=None, target_label=None):
        dataset = fetch_20newsgroups(subset='train')
        groups = [
            [
                'comp.graphics',
                'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware',
                'comp.windows.x',
            ],
            ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
            ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
            ['misc.forsale'],
            ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
            ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'],
        ]

        def flatten(list):
            return [item for sublist in list for item in sublist]

        label_list = dataset['target_names']
        label = []
        for _ in dataset['target']:
            _ = label_list[_]
            if _ not in flatten(groups):
                raise NotImplementedError

            for i, g in enumerate(groups):
                if _ in g:
                    label.append(i)
                    break
        label = np.array(label)
        print('Number of labels', len(label))
        idx_n = np.where(label == target_label)[0]
        idx_a = np.where(label != target_label)[0]
        label[idx_n] = 0
        label[idx_a] = 1
        # subsample
        if int(subsample * 0.95) > sum(label == 0):
            pts_n = sum(label == 0)
            pts_a = int(0.05 * pts_n / 0.95)
        else:
            pts_n = int(subsample * 0.95)
            pts_a = int(subsample * 0.05)

        idx_n = np.random.choice(idx_n, pts_n, replace=False)
        idx_a = np.random.choice(idx_a, pts_a, replace=False)
        idx = np.append(idx_n, idx_a)
        np.random.shuffle(idx)

        text = [dataset['data'][i] for i in idx]
        label = label[idx]
        del dataset

        text = [_.strip().replace('<br />', '') for _ in text]

        print(
            f'number of normal samples: {sum(label==0)}, number of anomalies: {sum(label==1)}'
        )

        return text, label

    # target_label = int(dataset_name.split('-')[1])
    text, label = data_generator(subsample=10000, target_label=0)
    raise NotImplementedError('Mmm not working as expected')
    # y = label
    # X = pd.DataFrame(data = text, columns = ['text'])


if __name__ == '__main__':
    main()
