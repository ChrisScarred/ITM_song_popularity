# Changelog ITM-song-popularity version 2021

This changelog documents the changes applied to the previous version of this project, located in the branch `2020`.

## 10. 04. 2021

### Added

- this `CHANGELOG.md`
- `requirements.txt` and information about their usage in `README.md`
- docstrings for all functions
- explicit typing for all functions
- a pydantic Song model

### Changed

- code refactoring - renaming
- the source codes were moved into the directory `src`
- the client id and the client secret are now supplied as environmental variables

### Removed

- following data from the previous version were removed:
  - the report
  - the result log
  - the entire folder `plots`
  - the entire folder `summaries`
  - the entire folder `data`
  - the entire folder `database`
- all the analysis scripts were removed, currently only data getters are implemented
