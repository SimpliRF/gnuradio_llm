# GNU Radio Large Language Model (LLM)

This project provides an LLM pipeline for quickly adding examples and re-tune
the foundation model of choice using LoRA (Low-Rank Adaption of LLMs).

The project also includes a dataset logger for GNU Radio Companion (GRC)
for emitting history-style training examples. GRC is often used for
education, so it makes sense to use it to create training data.
The dataset logger hooks into GRC and any flowgraphs deployed through the GRC
GUI to record traces that can then be used to create a dataset.
This dataset can in turn be used to fine-tune a model further.

**NOTE**: This project is not affiliated with or sponsored by any of the
official GNU Radio organizations.

## Building (Linux)

First, download the [conda-forge](https://conda-forge.org/download/)
installer and then install it using:
```
bash Miniforge3-$(uname)-$(uname -m).sh
```

From the project root, run the following commands:
```
conda env create -f environment.yml
conda activate gnuradio-llm
```

Then execute all scripts or tools from the root of the project.

## Usage (Linux)

To start up the radio CLI, simply run the following command in the project root:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
./app/radio_cli.py
```

Creating training examples is possible with the `grc_dataset_logger`. The tool
applies a patch and then every meaningful edit becomes a structured
prompt-completion pair. These pairs are grouped into histories and periodically
written to JSON dataset files to feed the tuning process.

To launch GRC with the logger enabled, simply run the following script:
```
./grc_dataset_logger/launch_grc.py
```

## GPL License
```
Copyright (c) 2025 SimpliRF, LLC.

GNU Radio LLM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GNU Radio LLM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Radio LLM.  If not, see <https://www.gnu.org/licenses/>.
```
