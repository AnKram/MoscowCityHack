import pickle
import pandas as pd
from pprint import pprint
import numpy as np
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import random
import sklearn
import random
import time
import string as _string

import pymorphy2 as pm
from sentence_transformers import SentenceTransformer
from parse_hh_data import download, parse

from const import N_CLUSTERS_FOR_SIMILARITY, N_IDS_PER_CLUSTER_FOR_SIMILARITY, N_CLUSTERS, MAX_ENG_SHARE

model = SentenceTransformer('gdrive/MyDrive/coding/classification_telegram/distiluse-base-multilingual-cased-v2_cpu', device='cpu')
morph = pm.MorphAnalyzer()

TO_REPLACE = ['<p>', '<em>', '<br />', '<ul>', '</p>', '</li>', '<li>', '</strong>', '<strong>', '</ul>', '</em>', '</ol>', '<ol>']
LETTERS_ENG = 'abcdefghijklmnopqrstuvwzyx +#&/'
LETTERS_ENG += LETTERS_ENG.upper()
LETTERS_RUS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '
LETTERS_RUS += LETTERS_RUS.upper()

POS_EXCLUDED = ['PREP', 'NPRO', 'CONJ', 'PRCL']
LENGTH = 3 * 23

SKILLS = set([i.lower() for i in pd.read_csv('key_skills.csv')['skill_name']])
COURSES = list(pd.read_csv('courses.csv')['course_name'])

df_v = pd.read_pickle('vacancies_1.pickle')


with open('xgbr2.pickle', 'rb') as f:
    xgbr = pickle.load(f)


with open('COURSES_VECTORS.pickle', 'rb') as f:
    courses_vectors = pickle.load(f)


def parse_key_skills(lst):
    return ' '.join([i['name'] for i in lst])


def parse_specializations(lst):
    return ' '.join([f"{i['name']} {i['profarea_name']}" for i in lst])


def clean_string(string):
    for i in TO_REPLACE:
        string = string.replace(i, ' ')

    string = remove_extra_spaces(string)

    return string


def parse_description(string):
    # string = string.replace('junior', ' ').replace('middle', ' ')
    lst = string.split()
    lst = [i for i in lst if i != 'и']
    string = ' '.join(lst)
    string = clean_string(string)
    return string


def remove_extra_spaces(string):
    lst = string.split()
    string = ' '.join(lst)
    return string


def get_eng(string):
    string = ''.join([i for i in string if i in LETTERS_ENG])
    lst = string.split()
    lst = [i for i in lst if len(i) >= 1 or i.lower() == 'c']
    string = ' '.join(lst)
    return string


def get_rus(string):
    string = ''.join([i for i in string if i in LETTERS_RUS])
    return string


def get_part_of_speech(string):
    return morph.parse(string)[0].tag.POS


def get_main(string):
    return ' '.join(string.split()[:LENGTH])


def remove_POS(string):
    lst = string.split()
    lst = [i for i in lst if get_part_of_speech(i) not in POS_EXCLUDED]
    string = ' '.join(lst)
    return string


def get_vacancy_eng_share(string):
    if len(string) == 0:
        return 1

    eng = get_eng(string)
    return len(eng) / len(string)


def get_all_skills(key_skills, eng):
    words_key_skills = key_skills.lower().split()
    words_eng = eng.lower().split()
    all_skills = set(words_key_skills) | set(words_eng)

    return ' '.join(all_skills)


def get_key_skills_from_eng(string):
    string = string.lower()
    lst = string.split()
    lst = [i for i in lst if i in SKILLS]
    return ' '.join(lst)


def get_vacancy_target_info(dct):
    key_skills = parse_key_skills(dct['key_skills'])
    key_skills = key_skills.replace('С++', 'C++')
    specializations = parse_specializations(dct['specializations'])
    description = parse_description(dct['description'])
    description = description.replace('С++', 'C++')

    name = dct['name']

    vacancy_data = dict()
    vacancy_data['id'] = dct['id']
    vacancy_data['string'] = f'{name} {key_skills} {description}'
    vacancy_data['experience'] = dct['experience']['name']
    vacancy_data['salary_our'] = '>160'
    vacancy_data['salary'] = dct['salary']
    vacancy_data['created_at'] = dct['created_at']
    vacancy_data['key_skills'] = key_skills
    vacancy_data['specializations'] = specializations
    vacancy_data['description'] = description
    vacancy_data['name'] = name

    vacancy_data['string'] = clean_string(vacancy_data['string'])
    vacancy_data['eng'] = get_eng(vacancy_data['string'])
    vacancy_data['eng'] = get_key_skills_from_eng(vacancy_data['eng'])  # ! replacing eng by skills
    vacancy_data['all_skills'] = get_all_skills(vacancy_data['key_skills'], vacancy_data['eng'])

    vacancy_data['rus'] = get_rus(vacancy_data['string'])
    vacancy_data['rus'] = remove_POS(vacancy_data['rus'])
    vacancy_data['rus_short'] = get_main(vacancy_data['rus'])

    return vacancy_data


def get_vector(sentence):
    return model.encode([sentence])[0]


def get_clusters(df_column, n_clusters):
    clustering_model = KMeans(n_clusters=n_clusters,
                              n_init=500,
                              max_iter=10000)

    clustering_model.fit(list(df_column))
    clusters = clustering_model.labels_
    return clusters


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_centroid(arrs):
    arrs = np.array(arrs)
    centroid = np.mean(arrs, axis=0)
    return centroid


def get_salary(dct):
    from_ = dct['from']
    to = dct['to']

    lst = [from_, to]
    lst = [i for i in lst if i]

    if len(lst) == 0:
        return ''

    salary = sum(lst) / len(lst)
    currency = dct['currency']

    if currency == 'USD':
        salary = salary * 72

    if currency == 'EUR':
        salary = salary * 87

    return salary


def get_train_test(length):
    lst = ['train'] * int(round(length * 0.8, 0))
    lst += ['test'] * (length - len(lst))
    random.shuffle(lst)
    return lst


def xgb_predict_salary(vector):
    return int(round(xgbr.predict(np.array([vector]))[0], 0))


def get_similarity(vec1, vec2):
    similarity = sklearn.metrics.pairwise.cosine_similarity([vec1], [vec2])[0][0]
    return similarity


def find_most_similar_id_by_vector(target_vector, series, ids):
    similarity_data = {get_similarity(target_vector, vector): vector for vector in series}
    most_similar_coef = max(similarity_data)
    most_similar_vector = similarity_data[most_similar_coef]
    for id_, vector in zip(ids, series):
        if list(vector) == list(most_similar_vector):
            return id_


def sort_dict(d, **kwargs):
    by = 'value'
    reverse = True

    if 'by' in kwargs:
        by = kwargs['by']

    if 'reverse' in kwargs:
        reverse = kwargs['reverse']

    if by == 'value':
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))
    if by == 'key':
        return dict(sorted(d.items(), key=lambda x: x[0], reverse=reverse))


def get_similarity_ids_by_vector(target_vector, series, ids):  # from high similarity to low
    similarity_data = {get_similarity(target_vector, vector): vector for vector in series}
    similarity_data = sort_dict(similarity_data, by='key', reverse=True)

    similarity_ids = list()
    for similarity_vector in similarity_data.values():
        for id_, vector in zip(ids, series):
            if list(vector) == list(similarity_vector):
                similarity_ids.append(id_)
                break

    return similarity_ids


def get_similarity_clusters_by_vector(target_vector, df):  # from high similarity to low
    df_ = df.copy()
    df_ = df_.drop_duplicates('cluster_all_skills')
    series = df_['centroid_all_skills']
    clusters = df_['cluster_all_skills']

    similarity_data = {get_similarity(target_vector, vector): vector for vector in series}
    similarity_data = sort_dict(similarity_data, by='key', reverse=True)

    similarity_ids = list()
    for similarity_vector in similarity_data.values():
        for cluster, vector in zip(clusters, series):
            if list(vector) == list(similarity_vector):
                similarity_ids.append(cluster)
                break

    return similarity_ids


def find_all_similar(cv_eng_vector, df):
    similarity_clusters = get_similarity_clusters_by_vector(cv_eng_vector, df)
    similarity_clusters = similarity_clusters[:N_CLUSTERS_FOR_SIMILARITY]
    similar_ids = dict()

    for cluster in similarity_clusters:
        df_ = df[df['cluster_all_skills'] == cluster]

        similarity_ids = get_similarity_ids_by_vector(cv_eng_vector, df_['all_skills_vector'], df_['id'])
        similarity_ids = similarity_ids[:N_IDS_PER_CLUSTER_FOR_SIMILARITY]

        similar_ids[cluster] = similarity_ids

    return similar_ids


def get_description_by_id(target_id, df):
    for id_, description in zip(df['id'], df['description']):
        if id_ == target_id:
            return clean_string(description)


def get_eng_by_id(target_id, df):
    for id_, eng in zip(df['id'], df['eng']):
        if id_ == target_id:
            return clean_string(eng)


def get_key_skills_by_id(target_id, df):
    for id_, key_skills in zip(df['id'], df['key_skills']):
        if id_ == target_id:
            return clean_string(key_skills)


def get_all_skills_by_id(target_id, df):
    for id_, all_skills in zip(df['id'], df['all_skills']):
        if id_ == target_id:
            return all_skills


def get_id_data(id_, df, cv):
    id_data = dict()
    description = get_description_by_id(id_, df)
    eng = get_eng_by_id(id_, df)
    key_skills = get_key_skills_by_id(id_, df)
    all_skills = get_all_skills_by_id(id_, df)

    id_data['description'] = description
    id_data['eng'] = eng
    id_data['key_skills'] = key_skills
    id_data['all_skills'] = all_skills
    new_skills = get_new_skills(cv, id_, df)
    id_data['course_recommended'] = find_closest_cource(courses_vectors, new_skills)

    return id_data


def fill_vacancies(data, df, cv):
    for cluster in data['vacancies']:
        for n, id_ in enumerate(data['vacancies'][cluster]):
            id_data = get_id_data(id_, df, cv)
            data['vacancies'][cluster][n] = id_data

    return data


def replace_punctuation(string):
    for i in ['.', ',']:
        string = string.replace(i, '')

    return string


def get_new_skills(cv, vacancy_id, df):
    words_cv = set(cv.lower().split())
    words_cv = set([replace_punctuation(i) for i in words_cv])

    all_skills = set(get_all_skills_by_id(vacancy_id, df).split())
    new_skills = (all_skills - words_cv) & SKILLS
    #print('all', all_skills)
    print(new_skills)
    #print(SKILLS)
    return ' '.join(new_skills)


def find_closest_cource(courses_vectors, new_skills):
    vector_new_skills = get_vector(new_skills)
    similarity_data = {get_similarity(vector_new_skills, vector): course for course, vector in courses_vectors.items()}
    most_similar_coef = max(similarity_data)
    most_similar_course = similarity_data[most_similar_coef]
    return most_similar_course


def get_bucker(integer):
    data = {
        'name': ['student', 'junior', 'pre mid', 'middle', 'pre senior', 'senior'],
        'min_sal': [-1, 50000, 75000, 150000, 220000, 350000],
        'max_sal': [50000, 75000, 150000, 220000, 350000, 99999999],
        'description': [
            'Вы в самом начале своего карьерного пути. Начните с общих курсов, чтобы определить своё направление развития',
            'Вы уже что-то умеете и не способны самостоятельно развиваться в рабочей команде.',
            'Вы уже не новичёк, но ещё не можете автономно лидировать задачи по разработке',
            'Вы полезный специалист в любой комаде. Продолжайте развиваться, что бы стать незаменимым специалистом в люой команде',
            'Ваши знания и опыт работы велики, но есть ещё непокорённые вершины в вашем направлении развития',
            'Поздравляем - вы сеньор-разработчик']
    }
    for name, min_sal, max_sal, desc in zip(data['name'], data['min_sal'], data['max_sal'], data['description']):
        if min_sal < integer <= max_sal:
            return {name: desc}

    return {'noname': 'молодец'}


def get_data(string):
    #vacancies_data = list()
    #mapping_centroids_eng = dict()

    # COURSES_VECTORS = {i: get_vector(i) for i in COURSES}

    # with open('COURSES_VECTORS.pickle', 'wb') as f:
    #    pickle.dump(COURSES_VECTORS, f)

    #for vacancy in df_v['vacancy']:
    #    vacancy_data = get_vacancy_target_info(vacancy)
    #    vacancies_data.append(vacancy_data)

    #df = pd.DataFrame(vacancies_data)

    #df['eng_share'] = df['description'].apply(get_vacancy_eng_share)
    #df = df[df['eng_share'] < MAX_ENG_SHARE]

    #df['all_skills_vector'] = df['all_skills'].apply(get_vector)
    #df['cluster_all_skills'] = get_clusters(df['all_skills_vector'], N_CLUSTERS)

    #for cluster, group in df.groupby('cluster_all_skills'):
    #    mapping_centroids_eng[cluster] = get_centroid(group['all_skills_vector'])

    #df['centroid_all_skills'] = [mapping_centroids_eng[i] for i in df['cluster_all_skills']]
    #df['salary_parsed'] = df['salary'].apply(get_salary)

    #string = 'Java kubernetes я знаю лучше всех'
    df = pd.read_pickle('df.pickle')
    cv_eng = get_eng(string)
    cv_eng_vector = get_vector(cv_eng)

    similarity_data = find_all_similar(cv_eng_vector, df)
    data = {'vacancies': similarity_data}
    data['salary'] = xgb_predict_salary(cv_eng_vector)
    data['bucket'] = get_bucker(data['salary'])
    data = fill_vacancies(data, df, cv_eng)

    return data


get_data('sql я знаю лучше всех')
