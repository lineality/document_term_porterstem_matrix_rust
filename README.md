# document_term_porterstem_matrix_rust

# Document-Term Porter-Stem Matrix in Rust
document_term_porterstem_matrix_rust

Basic, traditional, NLP utility for document term matrix creation,
for 'Production Data Science' in Rust.
- no-load reading and writing of potentially large files
- output of stems (word or sub-word ~tokens) per line in ~dictionary.txt
- for .csv input, .csv output as well

## Functionality:
1. Original single-word Porter Stemmer, output -> string
2. term matrix from multi-word document-string, output -> ~dictionary.txt
3. plain text utf8 document one-hot-matrix creation 
output -> document-term stem matrix as .csv and ~dictionary.txt
4. from .csv file one-hot-matrix creation 
without loading the whole .csv into memory
output -> document-term stem matrix as .csv and ~dictionary.txt
