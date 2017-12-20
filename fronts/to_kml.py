import csv
import simplekml

inputfile = csv.reader(open('prob_atlantic_1985-2010.csv','r'))
kml=simplekml.Kml()

for row in inputfile:
  kml.newpoint(prob=row[2], coords=[(row[0], row[1])])

kml.save('prob_atlantic_1985-2010.kml')

