from numpy import genfromtxt
import os
import pandas as pd
import numpy as np



class SantaDataLoad:

    def __init__(self):
        self.family_data=None
        self.total_nr_of_people=None
        self.total_nr_of_families = None

    def load_file(self, path):
        self.family_data = pd.read_csv(os.path.join(path, 'family_data.csv'), delimiter=',')
        return self.family_data

    def count_nr_of_people_attending(self):
        self.total_nr_of_people = np.sum(self.family_data['n_people'])
        return self.total_nr_of_people

    def count_nr_of_families(self):
        self.total_nr_of_families = len(self.family_data['family_id'])
        return self.total_nr_of_families

    def count_nr_of_days(self):
        choices_array = self.family_data.drop(['family_id'], axis=1)
        choices_array = choices_array.drop(['n_people'], axis=1)
        return np.max(np.unique(np.array(choices_array)))

    def starting_day(self):
        choices_array = self.family_data.drop(['family_id'], axis=1)
        choices_array = choices_array.drop(['n_people'], axis=1)
        return np.min(np.unique(np.array(choices_array)))

    def optimal_day_load(self):
        return self.total_nr_of_people / self.count_nr_of_days()

if __name__ == "__main__":
    data_load = SantaDataLoad()
    data_load.load_file("/Users/nicolaepetridean/jde/projects/titanic/try/santadata/")
    data_load.count_nr_of_families()
    data_load.count_nr_of_people_attending()
    print('total number of people is : ' + str(data_load.count_nr_of_people_attending()))
    print('total number of families is : ' + str(data_load.count_nr_of_families()))
    print('total number of days is : ' + str(data_load.count_nr_of_days()))
    print('starting day is : ' + str(data_load.starting_day()))
    print('optimal day load : ' + str(data_load.optimal_day_load()))
