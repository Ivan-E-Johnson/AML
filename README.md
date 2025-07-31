# Download Data 
1) This link will take you to the data download page: [Data Download](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230177)
Note you must Download the NBIA Data Retriever to download the data. [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)

2) Once you have the NBIA Data Retriever installed, you can download the data by opening the files that end in .tcia from the data download page.
3) Rename the top level folders to "OrigProstate" and "Segmentations" respectively.
4) Update the paths in [data_connector.py](Supervised/data_connector.py) to point to the location of the data on your machine.
5) Run the [data_connector.py](Supervised/data_connector.py) script to generate the data.csv file.