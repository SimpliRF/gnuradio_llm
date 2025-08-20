# GNU Radio Large Language Model (LLM)

This project provides an LLM pipeline for quickly adding examples and re-tune
the foundation model of choice using LoRA (Low-Rank Adaption of LLMs).

The project also includes a dataset logger for GNU Radio Companion (GRC)
for emitting history-style training examples.

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
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then execute all scripts or tools from the root of the project.

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
