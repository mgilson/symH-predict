### Sym-H prediction.

This is my capstone project for Udacity's Nanodegree program.  Fundamentally, it
is a Recurrent Neural Network (RNN) based Deep Neural Network (DNN) for
predicting future values in a multi-variate timeseries with missing data points.

It relies on the user having Tensorflow/Keras installed.

### Dataset

The dataset used is the OMNI dataset.  It contains Solar Wind plasma and
and magnetic field measurements as well as Sym-H and other Magnetospheric
activity indices.

For convenience, a truncated and compressed snapshot has been included as a
sqlite3 database.

To unzip the snapshot:

```sh
bunzip2 omni.db.bz2
```

For the most recent data and for fields that I have chosen to ignore, pull down
the files from [OMNIWeb's FTP server](ftp://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/).

If you like `sqlite3` better than working with CSVs, you can also use the helper
script to pack the OMNIWeb CSVs into a sqlite database:

```sh
tools/omni_csv_2_sqlite3.py /path/to/csvs/*
```


