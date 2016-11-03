import json
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from numpy import exp


def main():


    # Load data from json file
    with open('../data/Scary Monsters and Nice Sprites.json') as fp:
        analysis = json.load(fp)

    # Create 1048 x 12 array of pitches. One row for each segment, one column for each pitch
    pitchCoefficients = []
    for segment in analysis['segments']:
        pitchCoefficients.append(segment['pitches'])

    # Create square self-similarity matrix between pitches
    selfSimilarity = squareform(pdist(pitchCoefficients, metric='cosine'))

    selfSimilarity = 1 - selfSimilarity

    plt.imshow(selfSimilarity, cmap='hot')
    plt.clim(0, 1)

    # Title and Axis Labels
    plt.title('Pitch Self-similarity')
    plt.xlabel('Segments')
    plt.ylabel('Segments')
    plt.colorbar()

    plt.savefig('./files/pitch.png')


    # Clear the plot, so we can start over for Timbre
    plt.clf()


    # Create 1048 x 12 array of timbres, one row for each segment
    timbreCoefficients = []
    for segment2 in analysis['segments']:
        timbreCoefficients.append(segment2['timbre'])

    # Create square self-similarity matrix, use euclidean
    timbreSimilarity = squareform( pdist(timbreCoefficients))

    # Convert euclidean distances to a similarity measure
    timbreSimilarity = exp(-timbreSimilarity/100)

    plt.imshow(timbreSimilarity, cmap='hot')
    plt.clim(0, 1)

    # Title and Axis Labels
    plt.title('Timbre Self-similarity')
    plt.xlabel('Segments')
    plt.ylabel('Segments')
    plt.colorbar()

    plt.savefig('./files/timbre.png')

if __name__ == '__main__':
    main()
