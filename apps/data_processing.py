#import os
import numpy as np
import pandas as pd
import string
#from cs5710.apps import city_name_map as map
import re

city_name_map = {
    'Buirnsville': 'Burnsville',
    'Connelly': 'Connelly Springs',
    'Connelly Spring': 'Connelly Springs',
    'Connellys Springs': 'Connelly Springs',
    'Foresst City': 'Forest City',
    'Forest City, Nc 28043': 'Forest City',
    'Green Mtn': 'Green Mountain',
    'Green Mtn.': 'Green Mountain',
    'Henrrietta': 'Henrietta',
    'Mc Grady': 'McGrady',
    'Mcgrady': 'McGrady',
    'Mill Springs': 'Mill Spring',
    'Miller\'s Creek': 'Millers Creek',
    'Mooesboro': 'Mooresboro',
    'Moravianfalls': 'Moravian Falls',
    'Morgnton': 'Morganton',
    'Mroganton': 'Morganton',
    'N Wilkesboro': 'North Wilkesboro',
    'N. Wilkesboro': 'North Wilkesboro',
    'N.wilkesboro': 'North Wilkesboro',
    'North Wilesboro': 'North Wilkesboro',
    'North Wilkesbor': 'North Wilkesboro',
    'Noth Wilkesboro': 'North Wilkesboro',
    'Piney Crek': 'Piney Creek',
    'Robbinville': 'Robbinsville',
    'Ruthefordton': 'Rutherfordton',
    'Rutherford': 'Rutherfordton',
    'Wests Jefferson': 'West Jefferson',
    'Wilkeboro': 'Wilkesboro',
    'Wilkesboro, Nc 28697': 'Wilkesboro',
    'Wilkesboro28': 'Wilkesboro',
}

def clean_city(city):
    if isinstance(city, str):
        city = city.strip()
        city = string.capwords(city);

        if city.endswith(', Nc'):
            city = city.rstrip(city[-4:])

        if city.endswith(' Nc'):
            city = city.rstrip(city[-3:])

        #if city.endswith(', '):
         #   city = city.rstrip(city[-2:])

        #if city in city_name_map:
        #    city = city_name_map[city]

        city = city_name_map.get(city, city)

    return city


def clean_state(state):
    if isinstance(state, str):
        state = state.strip()
        state = state.upper()

        if state == 'NO':
            state = 'NC'

    return state

def clean_zip_code(zipCode):
    cleanedZip = ''

    if isinstance(zipCode, str):
        zipCode = zipCode.strip()

        #p = re.compile('[0-9]{5}')
        #match = p.match(zipCode)

        #if match:
        #    zipCode = match.group()
        #else:
        #    zipCode = np.nan

        # If the string is longer than five character, take the first five numbers
        if len(zipCode) >= 5:
            for charNum in range(len(zipCode)):
                if(len(cleanedZip) < 5 and zipCode[charNum].isdigit()):
                    cleanedZip += zipCode[charNum]
        else:
            cleanedZip = np.nan
    else:
        cleanedZip = zipCode

    return cleanedZip

def main():

    # Read basic_person.csv into a variable
    df = pd.read_csv('../data/basic_person.csv', index_col='acct_id_new')

    # Clean each of the columns
    df['city'] = df['city'].apply(clean_city)
    df['state'] = df['state'].apply(clean_state)
    df['zip'] = df['zip'].apply(clean_zip_code)

    # Make the 'files' directory if it doesn't exist
    #if not os.path.exists('files'):
        #os.makedirs('files')

    # Write the cleaned data out to files/cleaned.csv
    df.to_csv('files/cleaned.csv')

    ###### Part 2 - Joining Tables

    df1Again = pd.read_csv('../data/basic_person.csv', index_col='acct_id_new');
    df2 = pd.read_csv('../data/student_detail_v.csv', index_col='person_detail_id_new')
    df3 = pd.read_csv('../data/person_detail_f.csv', index_col='person_detail_id_new')

    df4 = df2.join(df3)

    df5 = df4.join(df1Again, on='acct_id_new')

    df5.to_csv('files/joined.csv')

if __name__ == '__main__':
    main()
