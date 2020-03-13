import os
import pickle
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.metrics.agreement import AnnotationTask

likertanno_criteria = ['Non-Redundancy', 'Referential Clarity', 'Grammaticality', 'Focus', 'Structure', 'Coherence', 'Readability',
         'Information Content', 'Spelling', 'Length', 'Overall Quality']
pairanno_criteria = ['Non-Redundancy', 'Referential Clarity', 'Structure', 'Readability', 'Information Content', 'Overall Quality']
all_methods_ordered = ['H1', 'H2', 'H3', 'H4', 'H5', 'LeadFirst', 'MMR', 'MMR*', 'Submodular', 'PG-MMR' ]

def likertanno_percentage_agreement(likertanno):
    # transforms the likertAnno dataset into
    likertanno.dropna(subset=['method'], inplace=True)
    likertanno = likertanno.astype({'topic': 'int32'}, )
    anno1_melted = pd.melt(likertanno, id_vars=['method', 'topic', 'annotator'], var_name='criterion', value_name ='score') #create one row for each method-annotator-topic-criterion combination. one annotation per row now
    annotators = list(set(likertanno.annotator))
    crowd_df = pd.DataFrame(columns=['method_i', 'method_j', 'criterion', 'topic', 'annotator', 'i greater j?'])
    to_concat = []
    for counter, annotator in enumerate(annotators):
        his_annotations = anno1_melted.loc[anno1_melted.annotator == annotator]
        his_annotations.set_index(['method', 'topic', 'annotator', 'criterion'], inplace=True)
        # his_annotations.set_index(['method', 'topic'], inplace=True)
        def row_compare(x, rowi):
            if x == rowi:
                return np.NaN
            return x > rowi
        iterlist = list(his_annotations['score'].iteritems())
        for i, (ind, score) in enumerate(iterlist):
            remaining_rows_by_annotator = list(list(zip(*iterlist))[0][i:])
            subresult = his_annotations['score'].loc[remaining_rows_by_annotator].apply(row_compare, args=[score]).rename('i greater j?')
            subresult = subresult.reset_index()
            subresult.rename(columns={'method': 'method_i'}, inplace=True)
            # topic: concatenation of both compared topics, always smaller one first, ie 1008_1044
            subresult['topic'] = subresult['topic'].map(lambda x: str(x) + '_' + str(ind[1]) if x < ind[1] else str(ind[1]) + '_' + str(x))
            subresult.dropna(inplace=True)
            subresult['method_j'] = ind[0]
            subresult = subresult.loc[(~(subresult.method_i == subresult.method_j)) & (subresult.criterion == ind[3])]
            to_concat.append(subresult)
        print('Annotator:', counter, annotator)
    crowd_df = pd.concat([crowd_df] + to_concat, ignore_index=False, sort=True)
    print(crowd_df.groupby(['annotator', 'criterion'])['i greater j?'].count().groupby('annotator').mean().rename('average number of comparisons. should be 40% to 75% of 49 choose 2==1179'))
    return crowd_df



def agreement_analysis(crowd_df, anno, ):
    assert crowd_df.groupby(['method_i', 'method_j', 'topic', 'criterion', 'annotator']).count().max().max() == 1
    crowd_df['i greater j?'] = crowd_df['i greater j?'].astype('int32')
    crowd_df.set_index(['method_i', 'method_j', 'topic', 'criterion', 'annotator'])
    # build comparisonId. topic is already ordered. we have to unify H1_H2 == H2_H1
    def methods_ordered(x, all_methods_ordered):
        mi, mj = x['method_i'], x['method_j']
        if all_methods_ordered.index(mi) < all_methods_ordered.index(mj):
            return mi + '_' + mj
        else:
            return mj + '_' + mi
    crowd_df['comparisonId'] = crowd_df.apply(methods_ordered, args=[all_methods_ordered], axis=1)
    crowd_df['comparisonId'] = crowd_df['comparisonId'] + '_' \
                                      + crowd_df['topic'].map(str) + '_' \
                                      + crowd_df['criterion']
    # crowd_df['i greater j?'] = crowd_df['i greater j?'].apply(lambda x: bool(random.getrandbits(1)))
    # grouping by comparisonId, not aggregating [method_i, method_j, topic, criterion], they are the same in each group
    weighted_voting_df = crowd_df.groupby(['comparisonId']).agg(
        votes=pd.NamedAgg(column='i greater j?', aggfunc='sum'),
        total_votes=pd.NamedAgg(column='i greater j?', aggfunc='count'),
        # comparisonId=pd.NamedAgg(column='comparisonId', aggfunc='first'), #no aggregation
        method_i = pd.NamedAgg(column='method_i', aggfunc='first'),
        method_j=pd.NamedAgg(column='method_j', aggfunc='first'),
        topic=pd.NamedAgg(column='topic', aggfunc='first'),
        criterion=pd.NamedAgg(column='criterion', aggfunc='first'),
    )
    weighted_voting_df = weighted_voting_df[weighted_voting_df.total_votes > 1]
    def percentage_agreement(row):
        if row['total_votes'] == 0:
            return np.NaN
        return row['votes'] / row['total_votes']
    perc_df = weighted_voting_df.apply(percentage_agreement, axis=1).rename('percentage_agreement').dropna()
    weighted_voting_df = weighted_voting_df.join(perc_df)
    weighted_voting_df = weighted_voting_df.reset_index()

    def won_vote(x):
        ag = x['percentage_agreement']
        if ag > 0.5:
            return True
        elif ag == 0.5:
            #for purposes of criteria agreement_analysis, its a 0.5 either way, so we just assign a random winner
            return bool(random.getrandbits(1))
        else: return False
    weighted_voting_df['left_won_vote?'] = weighted_voting_df.apply(won_vote, axis=1)
    comparisonId_df = crowd_df.copy(deep=True)
    method_i = weighted_voting_df.drop(columns='method_j').rename(columns={'method_i': 'method'})
    method_j = weighted_voting_df.drop(columns='method_i').rename(columns={'method_j': 'method'})
    method_j['left_won_vote?'] = ~method_j['left_won_vote?']
    method_j['percentage_agreement'] = method_j['percentage_agreement'].apply(lambda x: 1. - x)
    method_j['votes'] = method_j[['votes', 'total_votes']].apply(lambda x: (x['total_votes'] - x['votes']), axis=1)
    weighted_voting_df = pd.concat([method_i, method_j])
    weighted_voting_df.reset_index(drop=True)
    weighted_voting_df = weighted_voting_df.sort_values(['comparisonId'])
    assert 0.49 < weighted_voting_df['percentage_agreement'].mean() < 0.51

    #simple plots
    # by criterion (only take the winners, or the mean would be 0.5)
    critmeans = weighted_voting_df[weighted_voting_df['left_won_vote?']].groupby('criterion')['percentage_agreement'].mean()
    if anno == 'likertanno':
        critmeans.reindex(likertanno_criteria, axis=0)
        ax = critmeans.plot(y='criterion', x='percentage_agreement', kind='barh', )
        ax.set_xlim([0.6, 1.0])
    if anno == 'pairanno':
        critmeans.reindex(pairanno_criteria, axis=0)
        ax = critmeans.plot(y='criterion', x='percentage_agreement', kind='barh', )
        # ax.set_xlim([0.6, 1.0])
    ax.yaxis.label.set_visible(False)
    plt.tight_layout()
    plt.savefig('../figures/{}_percentage_agreement_criteria.pdf'.format(anno), bboxinches='tight', padinches=0)
    plt.show()
    #by methods
    sysmeans=weighted_voting_df.groupby('method')['percentage_agreement'].mean()
    ax = sysmeans.plot(kind='bar')
    ax.set_ylabel('mean percentage_agreement')
    ax.xaxis.label.set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/{}_percentage_agreement_by_method.pdf'.format(anno), bboxinches='tight', padinches=0)
    plt.show()

    ####################### Krippendorf alpha ##########
    #filter out single comparisons (only one annotator votes for a given comparison)
    filtered_comparisons = comparisonId_df.groupby('comparisonId').filter(lambda x: len(x) > 1)
    three_cols = ['annotator', 'comparisonId', 'i greater j?']
    task = AnnotationTask(data=filtered_comparisons[three_cols].values)

    krippendorf = [('Whole Dataset', task.alpha())]
    criteria = {'likertanno': likertanno_criteria, 'pairanno': pairanno_criteria}
    #by criteria:
    for criterion in criteria[anno][::-1]: # [::-1] reverses the list
        task = AnnotationTask(data=filtered_comparisons[
            filtered_comparisons.criterion == criterion][three_cols].values)
        # print('{} Krippendorf alpha for {}: \t{}'.format(criterion, anno, task.alpha()))
        krippendorf.append((criterion, task.alpha()))
    krippendorf = pd.DataFrame(data=krippendorf, columns=['criterion', 'krippendorf alpha'])
    ax = krippendorf.plot(kind='barh')
    ax.set_yticklabels(krippendorf.criterion)
    ax.set_xlabel('Krippendorf alpha')
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig('../figures/{}_krippendorf_agreement.pdf'.format(anno), bboxinches='tight', padinches=0)
    plt.show()
    return weighted_voting_df, crowd_df

# LIKERTANNO:
likertanno = pd.read_csv('../data/LikertAnno.csv')

if os.path.exists('../data/likertanno_as_pairwise.pickle'):
    with open('../data/likertanno_as_pairwise.pickle', 'rb') as f:
        likertanno_as_pairwise = pickle.load(f)
else:
    likertanno_as_pairwise = likertanno_percentage_agreement(likertanno)
with open('../data/likertanno_as_pairwise.pickle', 'wb') as f:
    pickle.dump(likertanno_as_pairwise, f)

likertanno_weighted_voting_df, likertanno_as_pairwise = agreement_analysis(likertanno_as_pairwise, 'likertanno')

# PAIRANNO

pairanno = pd.read_csv('../data/PairAnno.csv')
pairanno_weighted_voting_df, likertanno_as_pairwise = agreement_analysis(pairanno, 'pairanno')