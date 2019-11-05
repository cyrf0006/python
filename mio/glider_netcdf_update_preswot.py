import netCDF4


# Update attributes L0
dset = netCDF4.Dataset('SEA003_20180503_l0.nc', 'r+')
dset.abstract = 'Multi-disciplinary mission with deployment of the MIO SEA003 Glider along the satellite track Sentinel-3A 24 togheter with SOCIB 1000m-Slocum SDEEP00 glider (separate data file).'
dset.positioning_system='GPS'
dset.author_email='Frederic.Cyr@dfo-mpo.gc.ca'
dset.principal_investigator='Frederic Cyr and Andrea Doglioli'
dset.principal_investigator_email='Frederic.Cyr@dfo-mpo.gc.ca and andrea.doglioli@univ-amu.fr'
dset.acknowledgement = "CNES-Centre National d'Etudes Spatiales, France"
dset.citation = 'Pascual, Ananda ; Barcelo-Llull, Barbara ; Cutolo, Eugenio; Diaz-Barroso, Lara; Allen, John T.; Sanchez-Roman, Antonio; Alou-Font, Eva; Antich-Homar, Helena; Carbonero, Andrea; Casas, Benjamín ; Charcos, Miguel; Cyr, Frederic; Doglioli, Andrea M.; Fernandez, Juan Gabriel; Gomez Navarro, L.; Munoz, Cristian; Roque, David ; Ruiz, Inmaculada; Ruiz, Simon ; Ser-Giacomi, Enrico ; Torner, Marc; 2019; "Dataset from PRE-SWOT multi-platform experiment"; DIGITAL.CSIC; http://dx.doi.org/10.20350/digitalCSIC/8640'
dset.data_center = 'SOCIB Data Center'
dset.data_center_email = 'data.centre@socib.es'
dset.publisher_url='http://www.socib.eu/?seccion=dataCenter'
dset.institution = 'MIO-Mediterranean Institute of Oceanography, Marseille, France'
dset.institution_references = 'http://www.mio.univ-amu.fr/'
dset.project = 'BIOSWOT (PIs F.d\'Ovidio, A.Doglioli, G.Gregori)'
dset.publisher = 'MIO'
dset.publisher_url = 'http://www.mio.univ-amu.fr/'
dset.license = 'Approved for public release. Distribution Unlimited.' 
dset.close() # if you want to write the variable back to disk

# Update attributes L2
dset = netCDF4.Dataset('SEA003_20180503_l2.nc', 'r+')
dset.abstract = 'Multi-disciplinary mission with deployment of the MIO SEA003 Glider along the satellite track Sentinel-3A 24 togheter with SOCIB 1000m-Slocum SDEEP00 glider (separate data file).'
dset.positioning_system='GPS'
dset.author_email='Frederic.Cyr@dfo-mpo.gc.ca'
dset.principal_investigator='Frederic Cyr and Andrea Doglioli'
dset.principal_investigator_email='Frederic.Cyr@dfo-mpo.gc.ca and andrea.doglioli@univ-amu.fr'
dset.acknowledgement = "CNES-Centre National d'Etudes Spatiales, France."
dset.citation = 'Pascual, Ananda ; Barcelo-Llull, Barbara ; Cutolo, Eugenio; Diaz-Barroso, Lara; Allen, John T.; Sanchez-Roman, Antonio; Alou-Font, Eva; Antich-Homar, Helena; Carbonero, Andrea; Casas, Benjamín ; Charcos, Miguel; Cyr, Frederic; Doglioli, Andrea M.; Fernandez, Juan Gabriel; Gomez Navarro, L.; Munoz, Cristian; Roque, David ; Ruiz, Inmaculada; Ruiz, Simon ; Ser-Giacomi, Enrico ; Torner, Marc; 2019; "Dataset from PRE-SWOT multi-platform experiment"; DIGITAL.CSIC; http://dx.doi.org/10.20350/digitalCSIC/8640'
dset.data_center = 'SOCIB Data Center'
dset.data_center_email = 'data.centre@socib.es'
dset.publisher_url='http://www.socib.eu/?seccion=dataCenter'
dset.institution = 'MIO-Mediterranean Institute of Oceanography, Marseille, France'
dset.institution_references = 'http://www.mio.univ-amu.fr/'
dset.project = 'BIOSWOT (PIs F.d\'Ovidio, A.Doglioli, G.Gregori)'
dset.publisher = 'MIO'
dset.publisher_url = 'http://www.mio.univ-amu.fr/'
dset.license = 'Approved for public release. Distribution Unlimited.' 
dset.close() # if you want to write the variable back to disk

