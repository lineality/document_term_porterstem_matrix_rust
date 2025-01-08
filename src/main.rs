//! # Document-Term Porter-Stem Matrix in Rust
//! document_term_porterstem_matrix_rust
//!
//! 1. Original single-word Porter Stemmer, output -> string
//! 3. from multi-word document-string, output -> ?
//! 3. plain text utf8 document one-hot-matrix creation 
//! output -> document-term stem matrix as .csv
//! 4. from .csv file one-hot-matrix creation 
//! without loading the whole .csv into memory
//! output -> document-term stem matrix as .csv
//!
//! Frequency is likely not needed in many cases..
//! 
//! This is a Rust implementation of the Porter Stemming algorithm, based on the original
//! work by Martin Porter (1980). The algorithm reduces English words to their word stem,
//! base, or root form through a series of systematic steps.
//!
//! to read stem_dict.txt in python: 
//! ```python
//! # Read stems from dictionary file
//! with open('stem_dictionary.txt', 'r') as f:
//!     stems = [line.strip() for line in f]
//!
//! print("Stems:", stems)
//!
//! # If you need to read both the stems and the BOW matrix:
//! import pandas as pd
//!
//! # Read stems
//! with open('stem_dictionary.txt', 'r') as f:
//!     stems = [line.strip() for line in f]
//!
//! # Read BOW matrix
//! bow_df = pd.read_csv('output_with_bow.csv')
//!
//! # Verify columns match stems
//! stem_columns = [f"stem_{stem}" for stem in stems]
//! print("Stem columns present:", all(col in bow_df.columns for col in stem_columns))
//! ```
//!
//!
//! specifically based on cannonical ansi c version by Martin Porter 
//! at https://tartarus.org/martin/PorterStemmer/c.txt
//! 
//! ## Algorithm Overview
//! The Porter Stemmer follows five steps to reduce words to their stems:
//! 1. Handles plurals and past participles
//! 2. Handles various suffixes
//! 3. Deals with -ic-, -full, -ness etc.
//! 4. Handles -ant, -ence, etc.
//! 5. Removes final -e and changes -ll to -l in specific contexts
//! 
//! ## Reference
//! Porter, M.F., "An algorithm for suffix stripping", Program, Vol. 14,
//! No. 3, pp 130-137, July 1980.
//! 
//! ## Usage Example
//! ```rust
//! let mut stemmer = PorterStemmer::new();
//! assert_eq!(stemmer.stem("running"), "run");
//! assert_eq!(stemmer.stem("capabilities"), "capabl");
//! ```
//! 
//! ## Implementation Notes
//! - This implementation operates on lowercase ASCII characters only
//! - Input should be pre-processed to remove non-alphabetic characters
//! - The algorithm never increases word length
//! - Words of length 1 or 2 are not stemmed
//! 
//! ## Safety and Performance
//! - Memory safe: Uses Rust's Vec<char> instead of raw character buffers
//! - No unsafe blocks
//! - No external dependencies
//! - Maintains O(n) time complexity where n is word length
//!
//! From https://tartarus.org/martin/PorterStemmer/c.txt
//! /* This is the Porter stemming algorithm, coded up in ANSI C by the
//!    author. It may be be regarded as canonical, in that it follows the
//!    algorithm presented in
//!
//!    Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
//!    no. 3, pp 130-137,
//!
//!    only differing from it at the points marked --DEPARTURE-- below.
//!
//!    See also http://www.tartarus.org/~martin/PorterStemmer
//!
//!    The algorithm as described in the paper could be exactly replicated
//!    by adjusting the points of DEPARTURE, but this is barely necessary,
//!    because (a) the points of DEPARTURE are definitely improvements, and
//!    (b) no encoding of the Porter stemmer I have seen is anything like
//!    as exact as this version, even with the points of DEPARTURE!
//!
//!    You can compile it on Unix with 'gcc -O3 -o stem stem.c' after which
//!    'stem' takes a list of inputs and sends the stemmed equivalent to
//!    stdout.
//!
//!    The algorithm as encoded here is particularly fast.
//!
//!    Release 1: was many years ago
//!    Release 2: 11 Apr 2013
//!        fixes a bug noted by Matt Patenaude <matt@mattpatenaude.com>,
//!
//!        case 'o': if (ends("\03" "ion") && (b[j] == 's' || b[j] == 't')) break;
//!            ==>
//!        case 'o': if (ends("\03" "ion") && j >= k0 && (b[j] == 's' || b[j] == 't')) break;
//!
//!        to avoid accessing b[k0-1] when the word in b is "ion".
//!    Release 3: 25 Mar 2014
//!        fixes a similar bug noted by Klemens Baum <klemensbaum@gmail.com>,
//!        that if step1ab leaves a one letter result (ied -> i, aing -> a etc),
//!        step2 and step4 access the byte before the first letter. So we skip
//!        steps after step1ab unless k > k0.
//! */
//!
//! #include <string.h>  /* for memmove */
//!
//! #define TRUE 1
//! #define FALSE 0
//!
//! /* The main part of the stemming algorithm starts here. b is a buffer
//!    holding a word to be stemmed. The letters are in b[k0], b[k0+1] ...
//!    ending at b[k]. In fact k0 = 0 in this demo program. k is readjusted
//!    downwards as the stemming progresses. Zero termination is not in fact
//!    used in the algorithm.
//!
//!    Note that only lower case sequences are stemmed. Forcing to lower case
//!    should be done before stem(...) is called.
//! */
//!
//! static char * b;       /* buffer for word to be stemmed */
//! static int k,k0,j;     /* j is a general offset into the string */
//!
//! /* cons(i) is TRUE <=> b[i] is a consonant. */
//!
//! static int cons(int i)
//! {  switch (b[i])
//!    {  case 'a': case 'e': case 'i': case 'o': case 'u': return FALSE;
//!       case 'y': return (i==k0) ? TRUE : !cons(i-1);
//!       default: return TRUE;
//!    }
//! }
//!
//! /* m() measures the number of consonant sequences between k0 and j. if c is
//!    a consonant sequence and v a vowel sequence, and <..> indicates arbitrary
//!    presence,
//!
//!       <c><v>       gives 0
//!       <c>vc<v>     gives 1
//!       <c>vcvc<v>   gives 2
//!       <c>vcvcvc<v> gives 3
//!       ....
//! */
//!
//! static int m()
//! {  int n = 0;
//!    int i = k0;
//!    while(TRUE)
//!    {  if (i > j) return n;
//!       if (! cons(i)) break; i++;
//!    }
//!    i++;
//!    while(TRUE)
//!    {  while(TRUE)
//!       {  if (i > j) return n;
//!             if (cons(i)) break;
//!             i++;
//!       }
//!       i++;
//!       n++;
//!       while(TRUE)
//!       {  if (i > j) return n;
//!          if (! cons(i)) break;
//!          i++;
//!       }
//!       i++;
//!    }
//! }
//!
//! /* vowelinstem() is TRUE <=> k0,...j contains a vowel */
//!
//! static int vowelinstem()
//! {  int i; for (i = k0; i <= j; i++) if (! cons(i)) return TRUE;
//!    return FALSE;
//! }
//!
//! /* doublec(j) is TRUE <=> j,(j-1) contain a double consonant. */
//!
//! static int doublec(int j)
//! {  if (j < k0+1) return FALSE;
//!    if (b[j] != b[j-1]) return FALSE;
//!    return cons(j);
//! }
//!
//! /* cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
//!    and also if the second c is not w,x or y. this is used when trying to
//!    restore an e at the end of a short word. e.g.
//!
//!       cav(e), lov(e), hop(e), crim(e), but
//!       snow, box, tray.
//!
//! */
//!
//! static int cvc(int i)
//! {  if (i < k0+2 || !cons(i) || cons(i-1) || !cons(i-2)) return FALSE;
//!    {  int ch = b[i];
//!       if (ch == 'w' || ch == 'x' || ch == 'y') return FALSE;
//!    }
//!    return TRUE;
//! }
//!
//! /* ends(s) is TRUE <=> k0,...k ends with the string s. */
//!
//! static int ends(char * s)
//! {  int length = s[0];
//!    if (s[length] != b[k]) return FALSE; /* tiny speed-up */
//!    if (length > k-k0+1) return FALSE;
//!    if (memcmp(b+k-length+1,s+1,length) != 0) return FALSE;
//!    j = k-length;
//!    return TRUE;
//! }
//!
//! /* setto(s) sets (j+1),...k to the characters in the string s, readjusting
//!    k. */
//!
//! static void setto(char * s)
//! {  int length = s[0];
//!    memmove(b+j+1,s+1,length);
//!    k = j+length;
//! }
//!
//! /* r(s) is used further down. */
//!
//! static void r(char * s) { if (m() > 0) setto(s); }
//!
//! /* step1ab() gets rid of plurals and -ed or -ing. e.g.
//!
//!        caresses  ->  caress
//!        ponies    ->  poni
//!        ties      ->  ti
//!        caress    ->  caress
//!        cats      ->  cat
//!
//!        feed      ->  feed
//!        agreed    ->  agree
//!        disabled  ->  disable
//!
//!        matting   ->  mat
//!        mating    ->  mate
//!        meeting   ->  meet
//!        milling   ->  mill
//!        messing   ->  mess
//!
//!        meetings  ->  meet
//!
//! */
//!
//! static void step1ab()
//! {  if (b[k] == 's')
//!    {  if (ends("\04" "sses")) k -= 2; else
//!       if (ends("\03" "ies")) setto("\01" "i"); else
//!       if (b[k-1] != 's') k--;
//!    }
//!    if (ends("\03" "eed")) { if (m() > 0) k--; } else
//!    if ((ends("\02" "ed") || ends("\03" "ing")) && vowelinstem())
//!    {  k = j;
//!       if (ends("\02" "at")) setto("\03" "ate"); else
//!       if (ends("\02" "bl")) setto("\03" "ble"); else
//!       if (ends("\02" "iz")) setto("\03" "ize"); else
//!       if (doublec(k))
//!       {  k--;
//!          {  int ch = b[k];
//!             if (ch == 'l' || ch == 's' || ch == 'z') k++;
//!          }
//!       }
//!       else if (m() == 1 && cvc(k)) setto("\01" "e");
//!    }
//! }
//!
//! /* step1c() turns terminal y to i when there is another vowel in the stem. */
//!
//! static void step1c() { if (ends("\01" "y") && vowelinstem()) b[k] = 'i'; }
//!
//!
//! /* step2() maps double suffices to single ones. so -ization ( = -ize plus
//!    -ation) maps to -ize etc. note that the string before the suffix must give
//!    m() > 0. */
//!
//! static void step2() { switch (b[k-1])
//! {
//!     case 'a': if (ends("\07" "ational")) { r("\03" "ate"); break; }
//!               if (ends("\06" "tional")) { r("\04" "tion"); break; }
//!               break;
//!     case 'c': if (ends("\04" "enci")) { r("\04" "ence"); break; }
//!               if (ends("\04" "anci")) { r("\04" "ance"); break; }
//!               break;
//!     case 'e': if (ends("\04" "izer")) { r("\03" "ize"); break; }
//!               break;
//!     case 'l': if (ends("\03" "bli")) { r("\03" "ble"); break; } /*-DEPARTURE-*/
//!
//!  /* To match the published algorithm, replace this line with
//!     case 'l': if (ends("\04" "abli")) { r("\04" "able"); break; } */
//!
//!               if (ends("\04" "alli")) { r("\02" "al"); break; }
//!               if (ends("\05" "entli")) { r("\03" "ent"); break; }
//!               if (ends("\03" "eli")) { r("\01" "e"); break; }
//!               if (ends("\05" "ousli")) { r("\03" "ous"); break; }
//!               break;
//!     case 'o': if (ends("\07" "ization")) { r("\03" "ize"); break; }
//!               if (ends("\05" "ation")) { r("\03" "ate"); break; }
//!               if (ends("\04" "ator")) { r("\03" "ate"); break; }
//!               break;
//!     case 's': if (ends("\05" "alism")) { r("\02" "al"); break; }
//!               if (ends("\07" "iveness")) { r("\03" "ive"); break; }
//!               if (ends("\07" "fulness")) { r("\03" "ful"); break; }
//!               if (ends("\07" "ousness")) { r("\03" "ous"); break; }
//!               break;
//!     case 't': if (ends("\05" "aliti")) { r("\02" "al"); break; }
//!               if (ends("\05" "iviti")) { r("\03" "ive"); break; }
//!               if (ends("\06" "biliti")) { r("\03" "ble"); break; }
//!               break;
//!     case 'g': if (ends("\04" "logi")) { r("\03" "log"); break; } /*-DEPARTURE-*/
//!
//!  /* To match the published algorithm, delete this line */
//!
//! } }
//!
//! /* step3() deals with -ic-, -full, -ness etc. similar strategy to step2. */
//!
//! static void step3() { switch (b[k])
//! {
//!     case 'e': if (ends("\05" "icate")) { r("\02" "ic"); break; }
//!               if (ends("\05" "ative")) { r("\00" ""); break; }
//!               if (ends("\05" "alize")) { r("\02" "al"); break; }
//!               break;
//!     case 'i': if (ends("\05" "iciti")) { r("\02" "ic"); break; }
//!               break;
//!     case 'l': if (ends("\04" "ical")) { r("\02" "ic"); break; }
//!               if (ends("\03" "ful")) { r("\00" ""); break; }
//!               break;
//!     case 's': if (ends("\04" "ness")) { r("\00" ""); break; }
//!               break;
//! } }
//!
//! /* step4() takes off -ant, -ence etc., in context <c>vcvc<v>. */
//!
//! static void step4()
//! {  switch (b[k-1])
//!     {  case 'a': if (ends("\02" "al")) break; return;
//!        case 'c': if (ends("\04" "ance")) break;
//!                  if (ends("\04" "ence")) break; return;
//!        case 'e': if (ends("\02" "er")) break; return;
//!        case 'i': if (ends("\02" "ic")) break; return;
//!        case 'l': if (ends("\04" "able")) break;
//!                  if (ends("\04" "ible")) break; return;
//!        case 'n': if (ends("\03" "ant")) break;
//!                  if (ends("\05" "ement")) break;
//!                  if (ends("\04" "ment")) break;
//!                  if (ends("\03" "ent")) break; return;
//!        case 'o': if (ends("\03" "ion") && j >= k0 && (b[j] == 's' || b[j] == 't')) break;
//!                  if (ends("\02" "ou")) break; return;
//!                  /* takes care of -ous */
//!        case 's': if (ends("\03" "ism")) break; return;
//!        case 't': if (ends("\03" "ate")) break;
//!                  if (ends("\03" "iti")) break; return;
//!        case 'u': if (ends("\03" "ous")) break; return;
//!        case 'v': if (ends("\03" "ive")) break; return;
//!        case 'z': if (ends("\03" "ize")) break; return;
//!        default: return;
//!     }
//!     if (m() > 1) k = j;
//! }
//!
//! /* step5() removes a final -e if m() > 1, and changes -ll to -l if
//!    m() > 1. */
//!
//! static void step5()
//! {  j = k;
//!    if (b[k] == 'e')
//!    {  int a = m();
//!       if (a > 1 || a == 1 && !cvc(k-1)) k--;
//!    }
//!    if (b[k] == 'l' && doublec(k) && m() > 1) k--;
//! }
//!
//! /* In stem(p,i,j), p is a char pointer, and the string to be stemmed is from
//!    p[i] to p[j] inclusive. Typically i is zero and j is the offset to the last
//!    character of a string, (p[j+1] == '\0'). The stemmer adjusts the
//!    characters p[i] ... p[j] and returns the new end-point of the string, k.
//!    Stemming never increases word length, so i <= k <= j. To turn the stemmer
//!    into a module, declare 'stem' as extern, and delete the remainder of this
//!    file.
//! */
//!
//! int stem(char * p, int i, int j)
//! {  b = p; k = j; k0 = i; /* copy the parameters into statics */
//!    if (k <= k0+1) return k; /*-DEPARTURE-*/
//!
//!    /* With this line, strings of length 1 or 2 don't go through the
//!       stemming process, although no mention is made of this in the
//!       published algorithm. Remove the line to match the published
//!       algorithm. */
//!
//!    step1ab();
//!    if (k > k0) {
//!        step1c(); step2(); step3(); step4(); step5();
//!    }
//!    return k;
//! }
//!
//! /*--------------------stemmer definition ends here------------------------*/
//!
//! #include <stdio.h>
//! #include <stdlib.h>      /* for malloc, free */
//! #include <ctype.h>       /* for isupper, islower, tolower */
//!
//! static char * s;         /* a char * (=string) pointer; passed into b above */
//!
//! #define INC 50           /* size units in which s is increased */
//! static int i_max = INC;  /* maximum offset in s */
//!
//! void increase_s()
//! {  i_max += INC;
//!    {  char * new_s = (char *) malloc(i_max+1);
//!       { int i; for (i = 0; i < i_max; i++) new_s[i] = s[i]; } /* copy across */
//!       free(s); s = new_s;
//!    }
//! }
//!
//! #define LETTER(ch) (isupper(ch) || islower(ch))
//!
//! static void stemfile(FILE * f)
//! {  while(TRUE)
//!    {  int ch = getc(f);
//!       if (ch == EOF) return;
//!       if (LETTER(ch))
//!       {  int i = 0;
//!          while(TRUE)
//!          {  if (i == i_max) increase_s();
//!
//!             ch = tolower(ch); /* forces lower case */
//!
//!             s[i] = ch; i++;
//!             ch = getc(f);
//!             if (!LETTER(ch)) { ungetc(ch,f); break; }
//!          }
//!          s[stem(s,0,i-1)+1] = 0;
//!          /* the previous line calls the stemmer and uses its result to
//!             zero-terminate the string in s */
//!          printf("%s",s);
//!       }
//!       else putchar(ch);
//!    }
//! }
//!
//! int main(int argc, char * argv[])
//! {  int i;
//!    s = (char *) malloc(i_max+1);
//!    for (i = 1; i < argc; i++)
//!    {  FILE * f = fopen(argv[i],"r");
//!       if (f == 0) { fprintf(stderr,"File %s not found\n",argv[i]); exit(1); }
//!       stemfile(f);
//!    }
//!    free(s);
//!    return 0;
//! }

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use rayon::prelude::*;
use std::sync::Arc;
use std::collections::{
    HashMap,
    HashSet,
};
use std::sync::Mutex;
// use std::path::Path;

/// Porter Stemmer struct that maintains the state during stemming operations
#[derive(Debug)]
pub struct PorterStemmer {
    /// Buffer holding the word being processed
    buffer: Vec<char>,
    /// Current end position in buffer
    k: usize,
    /// Start position in buffer (typically 0)
    k0: usize,
    /// General offset used in various operations
    j: usize,
}


impl PorterStemmer {
    /// Creates a new Porter Stemmer instance
    /// 
    /// # Returns
    /// A new `PorterStemmer` with empty buffer and initialized indices
    pub fn new() -> Self {
        PorterStemmer {
            buffer: Vec::new(),
            k: 0,
            k0: 0,
            j: 0,
        }
    }
    
    
    /// Write stem dictionary to a separate file
    fn write_stem_dictionary(
        &self,
        stems: &HashSet<String>,
        dict_path: &str,
    ) -> io::Result<()> {
        let mut dict_file = File::create(dict_path)?;
        for stem in stems {
            writeln!(dict_file, "{}", stem)?;
        }
        Ok(())
    }

    /// Read stem dictionary from file
    fn read_stem_dictionary(dict_path: &str) -> io::Result<Vec<String>> {
        let file = File::open(dict_path)?;
        let reader = BufReader::new(file);
        let stems: Vec<String> = reader
            .lines()
            .collect::<io::Result<Vec<String>>>()?;
        Ok(stems)
    }

    /// Process CSV in a streaming fashion
    pub fn noload_process_csvtobow_matrix_streaming(
        &self,
        input_path: &str,
        output_path: &str,
        dict_path: &str,
        text_column: usize,
        chunk_size: usize,
    ) -> io::Result<()> {
        // First pass: collect stems using streaming
        let unique_stems = self.collect_unique_stems_streaming(input_path, text_column)?;
        
        // Write stems dictionary
        self.write_stem_dictionary(&unique_stems, dict_path)?;
        
        // Convert to ordered Vec
        let stem_vec: Vec<String> = unique_stems.into_iter().collect();
        
        // Second pass: create BOW matrix streaming
        let input = BufReader::new(File::open(input_path)?);
        let mut output = File::create(output_path)?;
        
        // Write header
        let header: String = stem_vec
            .iter()
            .map(|stem| format!("stem_{}", stem))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(output, "{}", header)?;

        // Process lines in chunks
        let mut lines = input.lines();
        let mut buffer = Vec::with_capacity(chunk_size);
        
        // Skip header
        let _ = lines.next();
        
        while let Some(line) = lines.next() {
            let line = line?;
            buffer.push(line);
            
            if buffer.len() >= chunk_size {
                self.process_chunk(&buffer, &stem_vec, text_column, &mut output)?;
                buffer.clear();
            }
        }
        
        // Process remaining lines
        if !buffer.is_empty() {
            self.process_chunk(&buffer, &stem_vec, text_column, &mut output)?;
        }
        
        Ok(())
    }


    /// Process a chunk of lines
    fn process_chunk(
        &self,
        lines: &[String],
        stems: &[String],
        text_column: usize,
        output: &mut File,
    ) -> io::Result<()> {
        let mut stemmer = PorterStemmer::new();
        
        for line in lines {
            let fields: Vec<&str> = line.split(',').collect();
            let mut bow_vector = vec![0; stems.len()];
            
            if let Some(text) = fields.get(text_column) {
                let words = Self::extract_words(text);
                for word in words {
                    let stemmed = stemmer.stem(&word);
                    if let Some(index) = stems.iter().position(|s| s == &stemmed) {
                        bow_vector[index] = 1;
                    }
                }
            }
            
            let bow_str = bow_vector
                .iter()
                .map(|&n| n.to_string())
                .collect::<Vec<String>>()
                .join(",");
            writeln!(output, "{},{}", line, bow_str)?;
        }
        
        Ok(())
    }


    /// Collect unique stems using streaming
    fn collect_unique_stems_streaming(
        &self,
        input_path: &str,
        text_column: usize,
    ) -> io::Result<HashSet<String>> {
        let file = File::open(input_path)?;
        let reader = BufReader::new(file);
        let mut unique_stems = HashSet::new();
        let mut stemmer = PorterStemmer::new();
        
        // Skip header
        for (i, line_result) in reader.lines().enumerate() {
            if i == 0 { continue; }
            
            let line = line_result?;
            let fields: Vec<&str> = line.split(',').collect();
            
            if let Some(text) = fields.get(text_column) {
                let words = Self::extract_words(text);
                for word in words {
                    unique_stems.insert(stemmer.stem(&word));
                }
            }
        }
        
        Ok(unique_stems)
    }

    
    /// Process a CSV file and create a document-term stem matrix
    /// 
    /// # Arguments
    /// * `input_path` - Path to input CSV file
    /// * `output_path` - Path for output CSV file
    /// * `text_column` - Name or index of the column containing text to stem
    /// 
    /// # Returns
    /// Result indicating success or failure
    /// Process CSV and create both stem dictionary and BOW matrix
    pub fn process_csv_to_bow_matrix(
        &self,
        input_path: &str,
        output_path: &str,
        dict_path: &str,
        text_column: usize,
    ) -> io::Result<()> {
        // First pass: collect unique stems
        let unique_stems = self.collect_unique_stems(input_path, text_column)?;
        
        // Write stem dictionary
        self.write_stem_dictionary(&unique_stems, dict_path)?;
        
        // Convert HashSet to Vec for ordered access
        let stem_vec: Vec<String> = unique_stems.into_iter().collect();
        
        // Second pass: create the document-term matrix
        let input_file = File::open(input_path)?;
        let output_file = File::create(output_path)?;
        self.create_bow_matrix(
            input_file,
            output_file,
            text_column,
            &stem_vec,
        )?;
        
        Ok(())
    }

    /// Collect all unique stems from the input CSV
    fn collect_unique_stems(
        &self,
        input_path: &str,
        text_column: usize,
    ) -> io::Result<HashSet<String>> {
        let file = File::open(input_path)?;
        let reader = BufReader::new(file);
        let mut unique_stems = HashSet::new();
        let mut stemmer = PorterStemmer::new();

        // Skip header
        for (i, line_result) in reader.lines().enumerate() {
            if i == 0 { continue; }
            
            let line = line_result?;
            let fields: Vec<&str> = line.split(',').collect();
            
            if let Some(text) = fields.get(text_column) {
                let words = Self::extract_words(text);
                for word in words {
                    unique_stems.insert(stemmer.stem(&word));
                }
            }
        }

        Ok(unique_stems)
    }

    /// Create the document-term matrix with ordered stems
    fn create_bow_matrix(
        &self,
        input_file: File,
        mut output_file: File,
        text_column: usize,
        stems: &[String],
    ) -> io::Result<()> {
        let reader = BufReader::new(input_file);
        let mut stemmer = PorterStemmer::new();

        // Write header without extra comma
        let header: String = stems
            .iter()
            .map(|stem| format!("stem_{}", stem))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(output_file, "{}", header)?;

        // Process each line
        for (i, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            
            if i == 0 {
                writeln!(output_file, "{}", line)?;
                continue;
            }

            let mut bow_vector = vec![0; stems.len()];
            let fields: Vec<&str> = line.split(',').collect();
            
            if let Some(text) = fields.get(text_column) {
                let words = Self::extract_words(text);
                for word in words {
                    let stemmed = stemmer.stem(&word);
                    if let Some(index) = stems.iter().position(|s| s == &stemmed) {
                        bow_vector[index] = 1;
                    }
                }
            }

            // Write original line plus BOW vector without extra comma
            let bow_str: String = bow_vector
                .iter()
                .map(|&n| n.to_string())
                .collect::<Vec<String>>()
                .join(",");
            writeln!(output_file, "{},{}", line, bow_str)?;
        }

        Ok(())
    }

    
    
    /// Extracts words from a string, filtering out non-alphabetic characters
    /// 
    /// # Arguments
    /// * `text` - Input string to extract words from
    /// 
    /// # Returns
    /// Vector of lowercase words
    fn extract_words(text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter_map(|word| {
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>()
                    .to_lowercase();
                
                if cleaned.is_empty() {
                    None
                } else {
                    Some(cleaned)
                }
            })
            .collect()
    }
    
    /// Process a document file line by line and create stem dictionary
    /// 
    /// # Arguments
    /// * `input_path` - Path to input text file
    /// * `dict_path` - Path for output stem dictionary
    /// * `chunk_size` - Number of lines to process at once
    /// 
    /// # Returns
    /// Result containing HashSet of stems for immediate use if needed
    pub fn noload_process_filedocument_streaming(
        &self,
        input_path: &str,
        dict_path: &str,
        chunk_size: usize,
    ) -> io::Result<HashSet<String>> {
        let file = File::open(input_path)?;
        let reader = BufReader::new(file);
        let mut unique_stems = HashSet::new();
        let mut stemmer = PorterStemmer::new();
        let mut buffer = Vec::with_capacity(chunk_size);

        // Process lines in chunks
        for line_result in reader.lines() {
            let line = line_result?;
            buffer.push(line);

            if buffer.len() >= chunk_size {
                // Process chunk
                for text in &buffer {
                    let words = Self::extract_words(text);
                    for word in words {
                        unique_stems.insert(stemmer.stem(&word));
                    }
                }
                buffer.clear();
            }
        }

        // Process any remaining lines
        if !buffer.is_empty() {
            for text in &buffer {
                let words = Self::extract_words(text);
                for word in words {
                    unique_stems.insert(stemmer.stem(&word));
                }
            }
        }

        // Write stem dictionary
        self.write_stem_dictionary(&unique_stems, dict_path)?;

        Ok(unique_stems)
    }

    /// Process document with frequency counting, streaming style
    pub fn noload_process_documentfile_frequencies_streaming(
        &self,
        input_path: &str,
        dict_path: &str,
        freq_output_path: &str,
        chunk_size: usize,
    ) -> io::Result<HashMap<String, usize>> {
        let file = File::open(input_path)?;
        let reader = BufReader::new(file);
        let mut word_frequencies = HashMap::new();
        let mut stemmer = PorterStemmer::new();
        let mut buffer = Vec::with_capacity(chunk_size);

        // Process lines in chunks
        for line_result in reader.lines() {
            let line = line_result?;
            buffer.push(line);

            if buffer.len() >= chunk_size {
                // Process chunk
                for text in &buffer {
                    let words = Self::extract_words(text);
                    for word in words {
                        let stemmed = stemmer.stem(&word);
                        *word_frequencies.entry(stemmed).or_insert(0) += 1;
                    }
                }
                buffer.clear();
            }
        }

        // Process any remaining lines
        if !buffer.is_empty() {
            for text in &buffer {
                let words = Self::extract_words(text);
                for word in words {
                    let stemmed = stemmer.stem(&word);
                    *word_frequencies.entry(stemmed).or_insert(0) += 1;
                }
            }
        }

        // Write stem dictionary
        let unique_stems: HashSet<String> = word_frequencies.keys().cloned().collect();
        self.write_stem_dictionary(&unique_stems, dict_path)?;

        // Write frequencies to separate file
        let mut freq_file = File::create(freq_output_path)?;
        for (stem, freq) in &word_frequencies {
            writeln!(freq_file, "{},{}", stem, freq)?;
        }

        Ok(word_frequencies)
    }
    
    

    /// Process a document and stem all words
    /// 
    /// # Arguments
    /// * `filepath` - Path to the document to process
    /// 
    /// # Returns
    /// Result containing HashMap of original words to their stemmed versions
    /// 
    /// # Errors
    /// Returns io::Error if file operations fail
    /// Process a document and stem all words
    pub fn process_file_document(&self, filepath: &str) -> io::Result<HashMap<String, String>> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let stemmed_words = Arc::new(Mutex::new(HashMap::new()));

        reader.lines()
            .par_bridge()
            .try_for_each(|line_result| -> io::Result<()> {
                let line = line_result?;
                let words = Self::extract_words(&line);
                
                let mut local_stemmer = PorterStemmer::new();
                
                for word in words {
                    let stemmed = local_stemmer.stem(&word);
                    let mut map = stemmed_words.lock().unwrap();
                    map.entry(word).or_insert_with(|| stemmed);
                }
                
                Ok(())
            })?;

        // Extract the HashMap from Arc<Mutex<...>>
        Ok(Arc::try_unwrap(stemmed_words)
            .unwrap()
            .into_inner()
            .unwrap())
    }

    /// Process a document and return word frequency counts of stemmed words
    /// 
    /// # Arguments
    /// * `filepath` - Path to the document to process
    /// 
    /// # Returns
    /// Result containing HashMap of stemmed words to their frequency counts
    /// 
    /// # Errors
    /// Returns io::Error if file operations fail
    pub fn process_documentfile_with_frequencies(&self, filepath: &str) -> io::Result<HashMap<String, usize>> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let word_frequencies = Arc::new(Mutex::new(HashMap::new()));

        reader.lines()
            .par_bridge()
            .try_for_each(|line_result| -> io::Result<()> {
                let line = line_result?;
                let words = Self::extract_words(&line);
                
                let mut local_stemmer = PorterStemmer::new();
                
                for word in words {
                    let stemmed = local_stemmer.stem(&word);
                    let mut map = word_frequencies.lock().unwrap();
                    *map.entry(stemmed).or_insert(0) += 1;
                }
                
                Ok(())
            })?;

        Ok(Arc::try_unwrap(word_frequencies)
            .unwrap()
            .into_inner()
            .unwrap())
    }

    /// Process text content directly (without file I/O)
    /// 
    /// # Arguments
    /// * `content` - String content to process
    /// 
    /// # Returns
    /// HashMap of original words to their stemmed versions
    pub fn process_text(&self, content: &str) -> HashMap<String, String> {
        let stemmed_words = Arc::new(Mutex::new(HashMap::new()));
        
        content.lines()
            .par_bridge()
            .for_each(|line| {
                let words = Self::extract_words(line);
                let mut local_stemmer = PorterStemmer::new();
                
                for word in words {
                    let stemmed = local_stemmer.stem(&word);
                    let mut map = stemmed_words.lock().unwrap();
                    map.entry(word).or_insert_with(|| stemmed);
                }
            });

        Arc::try_unwrap(stemmed_words)
            .unwrap()
            .into_inner()
            .unwrap()
    }
    
    
    

    /// Determines if a character at position i is a consonant
    /// 
    /// # Arguments
    /// * `i` - Index in the buffer to check
    /// 
    /// # Returns
    /// * `true` if the character is a consonant
    /// * `false` if the character is a vowel
    /// 
    /// # Notes
    /// - A consonant is defined as any letter other than A, E, I, O, or U
    /// - Y is considered a consonant when:
    ///   1. It's the first letter (k0)
    ///   2. The previous letter is a consonan
    ///
    /// Returns true if the character at position i is a consonant
    fn is_consonant(&self, i: usize) -> bool {
        match self.buffer[i] {
            'a' | 'e' | 'i' | 'o' | 'u' => false,
            'y' => if i == self.k0 {
                true
            } else {
                !self.is_consonant(i - 1)
            },
            _ => true,
        }
    }

    /// Measures the number of consonant sequences between k0 and j
    /// 
    /// # Returns
    /// The number of consonant-vowel sequences (measure)
    /// 
    /// # Examples
    /// - TR.A gives measure 1
    /// - TRE.A gives measure 1
    /// - Y gives measure 0
    /// - BY gives measure 1
    /// 
    /// Where '.' indicates the current position
    fn measure(&self) -> usize {
        let mut n = 0;
        let mut i = self.k0;
        
        loop {
            if i > self.j { return n; }
            if !self.is_consonant(i) { break; }
            i += 1;
        }
        
        i += 1;
        
        loop {
            loop {
                if i > self.j { return n; }
                if self.is_consonant(i) { break; }
                i += 1;
            }
            
            i += 1;
            n += 1;
            
            loop {
                if i > self.j { return n; }
                if !self.is_consonant(i) { break; }
                i += 1;
            }
            
            i += 1;
        }
    }

    /// Returns true if k0,...j contains a vowel
    fn vowel_in_stem(&self) -> bool {
        (self.k0..=self.j).any(|i| !self.is_consonant(i))
    }

    /// Returns true if j,(j-1) contain a double consonant
    fn double_consonant(&self, j: usize) -> bool {
        if j < self.k0 + 1 { return false; }
        if self.buffer[j] != self.buffer[j-1] { return false; }
        self.is_consonant(j)
    }

    /// Returns true if i-2,i-1,i has the form consonant-vowel-consonant
    /// and also if the second c is not w,x or y
    fn cvc(&self, i: usize) -> bool {
        if i < self.k0 + 2 
            || !self.is_consonant(i)
            || self.is_consonant(i-1)
            || !self.is_consonant(i-2) {
            return false;
        }
        
        match self.buffer[i] {
            'w' | 'x' | 'y' => false,
            _ => true,
        }
    }

    /// Returns true if the word ends with the given string
    fn ends_with(&mut self, s: &str) -> bool {
        let length = s.len();
        if length > self.k - self.k0 + 1 { return false; }
        
        let end = &self.buffer[(self.k + 1 - length)..=self.k];
        let s_chars: Vec<char> = s.chars().collect();
        
        if end != &s_chars[..] { return false; }
        
        self.j = self.k - length;
        true
    }

    /// Sets (j+1),...k to the characters in the string s
    fn set_to(&mut self, s: &str) {
        let s_chars: Vec<char> = s.chars().collect();
        let length = s_chars.len();
        
        for (i, &ch) in s_chars.iter().enumerate() {
            self.buffer[self.j + 1 + i] = ch;
        }
        
        self.k = self.j + length;
    }

    /// Main stemming function that processes a word through all steps
    /// 
    /// # Arguments
    /// * `word` - Input word to be stemmed
    /// 
    /// # Returns
    /// The stemmed word as a String
    /// 
    /// # Examples
    /// ```
    /// let mut stemmer = PorterStemmer::new();
    /// assert_eq!(stemmer.stem("running"), "run");
    /// ```
    /// 
    /// # Process
    /// 1. Converts input to lowercase
    /// 2. Applies steps 1a through 5 in sequence
    /// 3. Returns the stemmed result
    pub fn stem(&mut self, word: &str) -> String {
        if word.is_empty() { return String::new(); }
        
        // Convert to lowercase and store in buffer
        self.buffer = word.to_lowercase().chars().collect();
        self.k = self.buffer.len() - 1;
        self.k0 = 0;
        
        if self.k <= self.k0 + 1 { 
            return self.buffer.iter().collect(); 
        }

        self.step1ab();
        if self.k > self.k0 {
            self.step1c();
            self.step2();
            self.step3();
            self.step4();
            self.step5();
        }

        self.buffer[0..=self.k].iter().collect()
    }
    
    /// Step 1ab handles plurals and past participles
    /// 
    /// # Transformations
    /// - SSES -> SS (caresses -> caress)
    /// - IES  -> I  (ponies -> poni)
    /// - SS   -> SS (caress -> caress)
    /// - S    ->    (cats -> cat)
    /// 
    /// And then:
    /// - (m>0) EED -> EE     (agreed -> agree)
    /// - (*v*) ED  ->        (plastered -> plaster)
    /// - (*v*) ING ->        (motoring -> motor)
    fn step1ab(&mut self) {
        if self.buffer[self.k] == 's' {
            if self.ends_with("sses") {
                self.k -= 2;
            } else if self.ends_with("ies") {
                self.set_to("i");
            } else if self.buffer[self.k - 1] != 's' {
                self.k -= 1;
            }
        }

        if self.ends_with("eed") {
            if self.measure() > 0 {
                self.k -= 1;
            }
        } else if (self.ends_with("ed") || self.ends_with("ing")) && self.vowel_in_stem() {
            self.k = self.j;

            if self.ends_with("at") {
                self.set_to("ate");
            } else if self.ends_with("bl") {
                self.set_to("ble");
            } else if self.ends_with("iz") {
                self.set_to("ize");
            } else if self.double_consonant(self.k) {
                self.k -= 1;
                let ch = self.buffer[self.k];
                if ch == 'l' || ch == 's' || ch == 'z' {
                    self.k += 1;
                }
            } else if self.measure() == 1 && self.cvc(self.k) {
                self.set_to("e");
            }
        }
    }

    /// Step 1c turns terminal y to i when there is another vowel in the stem
    /// 
    /// # Examples
    /// - happy -> happi
    /// - sky -> sky (unchanged)
    fn step1c(&mut self) {
        if self.ends_with("y") && self.vowel_in_stem() {
            self.buffer[self.k] = 'i';
        }
    }

    /// Step 2 maps double suffices to single ones when measure > 0
    /// 
    /// # Examples
    /// - ATIONAL -> ATE (relational -> relate)
    /// - TIONAL  -> TION (conditional -> condition)
    /// - ENCI    -> ENCE (valenci -> valence)
    fn step2(&mut self) {
        if self.k <= self.k0 { return; }
        
        match self.buffer[self.k - 1] {
            'a' => {
                if self.ends_with("ational") { self.replace_suffix_if_stem_measured("ate"); }
                else if self.ends_with("tional") { self.replace_suffix_if_stem_measured("tion"); }
            },
            'c' => {
                if self.ends_with("enci") { self.replace_suffix_if_stem_measured("ence"); }
                else if self.ends_with("anci") { self.replace_suffix_if_stem_measured("ance"); }
            },
            'e' => {
                if self.ends_with("izer") { self.replace_suffix_if_stem_measured("ize"); }
            },
            'l' => {
                if self.ends_with("bli") { self.replace_suffix_if_stem_measured("ble"); }
                else if self.ends_with("alli") { self.replace_suffix_if_stem_measured("al"); }
                else if self.ends_with("entli") { self.replace_suffix_if_stem_measured("ent"); }
                else if self.ends_with("eli") { self.replace_suffix_if_stem_measured("e"); }
                else if self.ends_with("ousli") { self.replace_suffix_if_stem_measured("ous"); }
            },
            'o' => {
                if self.ends_with("ization") { self.replace_suffix_if_stem_measured("ize"); }
                else if self.ends_with("ation") { self.replace_suffix_if_stem_measured("ate"); }
                else if self.ends_with("ator") { self.replace_suffix_if_stem_measured("ate"); }
            },
            's' => {
                if self.ends_with("alism") { self.replace_suffix_if_stem_measured("al"); }
                else if self.ends_with("iveness") { self.replace_suffix_if_stem_measured("ive"); }
                else if self.ends_with("fulness") { self.replace_suffix_if_stem_measured("ful"); }
                else if self.ends_with("ousness") { self.replace_suffix_if_stem_measured("ous"); }
            },
            't' => {
                if self.ends_with("aliti") { self.replace_suffix_if_stem_measured("al"); }
                else if self.ends_with("iviti") { self.replace_suffix_if_stem_measured("ive"); }
                else if self.ends_with("biliti") { self.replace_suffix_if_stem_measured("ble"); }
            },
            'g' => {
                if self.ends_with("logi") { self.replace_suffix_if_stem_measured("log"); }
            },
            _ => {}
        }
    }

    /// Step 3 deals with -ic-, -full, -ness etc.
    /// 
    /// # Examples
    /// - ICATE -> IC (triplicate -> triplic)
    /// - ATIVE ->    (formative -> form)
    /// - ALIZE -> AL (formalize -> formal)
    fn step3(&mut self) {
        match self.buffer[self.k] {
            'e' => {
                if self.ends_with("icate") { self.replace_suffix_if_stem_measured("ic"); }
                else if self.ends_with("ative") { self.replace_suffix_if_stem_measured(""); }
                else if self.ends_with("alize") { self.replace_suffix_if_stem_measured("al"); }
            },
            'i' => {
                if self.ends_with("iciti") { self.replace_suffix_if_stem_measured("ic"); }
            },
            'l' => {
                if self.ends_with("ical") { self.replace_suffix_if_stem_measured("ic"); }
                else if self.ends_with("ful") { self.replace_suffix_if_stem_measured(""); }
            },
            's' => {
                if self.ends_with("ness") { self.replace_suffix_if_stem_measured(""); }
            },
            _ => {}
        }
    }

    /// Step 4 removes suffixes when measure > 1
    /// 
    /// # Examples
    /// - AL    ->  (revival -> reviv)
    /// - ANCE  ->  (allowance -> allow)
    /// - ENCE  ->  (inference -> infer)
    fn step4(&mut self) {
        if self.k <= self.k0 { return; }

        match self.buffer[self.k - 1] {
            'a' => {
                if self.ends_with("al") {}
                else { return; }
            },
            'c' => {
                if self.ends_with("ance") {}
                else if self.ends_with("ence") {}
                else { return; }
            },
            'e' => {
                if self.ends_with("er") {}
                else { return; }
            },
            'i' => {
                if self.ends_with("ic") {}
                else { return; }
            },
            'l' => {
                if self.ends_with("able") {}
                else if self.ends_with("ible") {}
                else { return; }
            },
            'n' => {
                if self.ends_with("ant") {}
                else if self.ends_with("ement") {}
                else if self.ends_with("ment") {}
                else if self.ends_with("ent") {}
                else { return; }
            },
            'o' => {
                if self.ends_with("ion") && self.j >= self.k0 && 
                   (self.buffer[self.j] == 's' || self.buffer[self.j] == 't') {}
                else if self.ends_with("ou") {}
                else { return; }
            },
            's' => {
                if self.ends_with("ism") {}
                else { return; }
            },
            't' => {
                if self.ends_with("ate") {}
                else if self.ends_with("iti") {}
                else { return; }
            },
            'u' => {
                if self.ends_with("ous") {}
                else { return; }
            },
            'v' => {
                if self.ends_with("ive") {}
                else { return; }
            },
            'z' => {
                if self.ends_with("ize") {}
                else { return; }
            },
            _ => { return; }
        }
        if self.measure() > 1 {
            self.k = self.j;
        }
    }

    /// Step 5 removes final -e if measure > 1, and changes -ll to -l if measure > 1
    /// 
    /// # Examples
    /// - E     ->  (probate -> probat, rate -> rate)
    /// - L     ->  (controll -> control)
    fn step5(&mut self) {
        self.j = self.k;
        if self.buffer[self.k] == 'e' {
            let a = self.measure();
            if a > 1 || (a == 1 && !self.cvc(self.k - 1)) {
                self.k -= 1;
            }
        }
        if self.buffer[self.k] == 'l' && self.double_consonant(self.k) && self.measure() > 1 {
            self.k -= 1;
        }
    }

    /// Helper function for step2 and step3
    /// replaces current suffix with new_suffix if the stem has measure > 0
    fn replace_suffix_if_stem_measured(&mut self, s: &str) {
        if self.measure() > 0 {
            self.set_to(s);
        }
    }
}

// Test with bash: Cargo Test
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::write;
    use tempfile::NamedTempFile;
    // use super::*;

    #[test]
    fn test_basic_stemming() {
        let mut stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("caresses"), "caress");
        assert_eq!(stemmer.stem("ponies"), "poni");
        assert_eq!(stemmer.stem("ties"), "ti");
        assert_eq!(stemmer.stem("caress"), "caress");
        assert_eq!(stemmer.stem("cats"), "cat");
    }

    #[test]
    fn test_complex_stemming() {
        let mut stemmer = PorterStemmer::new();
        assert_eq!(stemmer.stem("troubled"), "troubl");
        assert_eq!(stemmer.stem("troubles"), "troubl");
        assert_eq!(stemmer.stem("troubling"), "troubl");
        assert_eq!(stemmer.stem("capability"), "capabl");
        assert_eq!(stemmer.stem("marketing"), "market");
    }


    #[test]
    fn test_extract_words() {
        let text = "Hello, World! This is a test.";
        let words = PorterStemmer::extract_words(text);
        assert_eq!(words, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_process_text() {
        let stemmer = PorterStemmer::new();
        let text = "running and jumping";
        let result = stemmer.process_text(text);
        
        assert_eq!(result.get("running").unwrap(), "run");
        assert_eq!(result.get("jumping").unwrap(), "jump");
    }

    #[test]
    fn test_process_document() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        write(&temp_file, "running and jumping\nwalking and talking")?;
        
        let stemmer = PorterStemmer::new();
        let result = stemmer.process_file_document(temp_file.path().to_str().unwrap())?;
        
        assert_eq!(result.get("running").unwrap(), "run");
        assert_eq!(result.get("jumping").unwrap(), "jump");
        assert_eq!(result.get("walking").unwrap(), "walk");
        assert_eq!(result.get("talking").unwrap(), "talk");
        
        Ok(())
    }
    

    #[test]
    fn test_document_streaming() -> io::Result<()> {
        let input_file = NamedTempFile::new()?;
        let dict_file = NamedTempFile::new()?;
        
        // Write test data
        write!(
            input_file.as_file(),
            "running and jumping\nwalking and talking"
        )?;
        input_file.as_file().sync_all()?;

        let stemmer = PorterStemmer::new();
        let stems = stemmer.noload_process_filedocument_streaming(
            input_file.path().to_str().unwrap(),
            dict_file.path().to_str().unwrap(),
            10,
        )?;

        assert!(stems.contains("run"));
        assert!(stems.contains("jump"));
        
        // Verify dictionary file
        let saved_stems = PorterStemmer::read_stem_dictionary(dict_file.path().to_str().unwrap())?;
        assert!(saved_stems.contains(&"run".to_string()));
        assert!(saved_stems.contains(&"jump".to_string()));

        Ok(())
    }

    #[test]
    fn test_document_frequencies_streaming() -> io::Result<()> {
        let input_file = NamedTempFile::new()?;
        let dict_file = NamedTempFile::new()?;
        let freq_file = NamedTempFile::new()?;
        
        // Write test data
        write!(
            input_file.as_file(),
            "running and jumping\nrunning and walking"
        )?;
        input_file.as_file().sync_all()?;

        let stemmer = PorterStemmer::new();
        let frequencies = stemmer.noload_process_documentfile_frequencies_streaming(
            input_file.path().to_str().unwrap(),
            dict_file.path().to_str().unwrap(),
            freq_file.path().to_str().unwrap(),
            10,
        )?;

        assert_eq!(frequencies.get("run"), Some(&2));  // "running" appears twice
        assert_eq!(frequencies.get("jump"), Some(&1));
        
        Ok(())
    }
    
    
    // use super::*;
    // use tempfile::NamedTempFile;

    // .csv
    #[test]
    fn test_csv_processing() -> io::Result<()> {
        // Create a temporary CSV file
        let input_file = NamedTempFile::new()?;
        write!(
            input_file.as_file(),
            "id,text\n1,running and jumping\n2,walking and talking"
        )?;

        let output_file = NamedTempFile::new()?;
        let bow_dict = NamedTempFile::new()?;
        
        let stemmer = PorterStemmer::new();
        stemmer.process_csv_to_bow_matrix(
            input_file.path().to_str().unwrap(),
            output_file.path().to_str().unwrap(),
            bow_dict.path().to_str().unwrap(),
            1,
        )?;

        // Read and verify output
        let reader = BufReader::new(File::open(output_file.path())?);
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
        
        assert!(lines.len() > 0);
        assert!(lines[0].contains("stem_run"));
        assert!(lines[0].contains("stem_jump"));
        
        Ok(())
    }
    
    #[test]
    fn test_csv_and_dictionary_processing() -> io::Result<()> {
        // Create temporary files
        let input_file = NamedTempFile::new()?;
        let output_file = NamedTempFile::new()?;
        let dict_file = NamedTempFile::new()?;
        
        // Write test data
        write!(
            input_file.as_file(),
            "id,text\n1,running and jumping\n2,walking and talking"
        )?;

        let stemmer = PorterStemmer::new();
        stemmer.process_csv_to_bow_matrix(
            input_file.path().to_str().unwrap(),
            output_file.path().to_str().unwrap(),
            dict_file.path().to_str().unwrap(),
            1,
        )?;

        // Verify dictionary
        let stems = PorterStemmer::read_stem_dictionary(dict_file.path().to_str().unwrap())?;
        assert!(stems.contains(&"run".to_string()));
        assert!(stems.contains(&"jump".to_string()));
        
        // Verify BOW matrix
        let reader = BufReader::new(File::open(output_file.path())?);
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
        assert!(lines.len() > 0);
        assert!(!lines[0].ends_with(','));  // No extra comma
        
        Ok(())
    }    

    #[test]
    fn test_noload_csv_and_dictionary_processing() -> io::Result<()> {
        // Create temporary files
        let input_file = NamedTempFile::new()?;
        let output_file = NamedTempFile::new()?;
        let dict_file = NamedTempFile::new()?;
        
        // Write test data
        write!(
            input_file.as_file(),
            "id,text\n1,running and jumping\n2,walking and talking"
        )?;
        
        // Verify test data was written
        input_file.as_file().sync_all()?;
        
        let stemmer = PorterStemmer::new();
        stemmer.noload_process_csvtobow_matrix_streaming(
            input_file.path().to_str().unwrap(),
            output_file.path().to_str().unwrap(),
            dict_file.path().to_str().unwrap(),
            1,  // text is in column 1, not 0
            10,
        )?;

        // Verify dictionary
        let stems = PorterStemmer::read_stem_dictionary(dict_file.path().to_str().unwrap())?;
        println!("Stems in dictionary: {:?}", stems);  // Debug print
        assert!(stems.contains(&"run".to_string()));
        assert!(stems.contains(&"jump".to_string()));
        
        // Verify BOW matrix
        let reader = BufReader::new(File::open(output_file.path())?);
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
        println!("Output lines: {:?}", lines);  // Debug print
        assert!(!lines.is_empty());
        assert!(!lines[0].ends_with(','));  // No extra comma
        
        Ok(())
    }
}  // End of Tests

fn main() -> io::Result<()> {
    
    //////////////////
    // 1. single word
    //////////////////
    print!("1. single word\n");
    let mut stemmer = PorterStemmer::new();
    let stemmed = stemmer.stem("running");
    println!("stemmed = stemmer.stem('running'): {}\n", stemmed); // Outputs: "run"
    
    ////////////////////////
    // 2. single doc string
    ////////////////////////
    // Process text directly
    print!("2. single doc string\n");
    let text = "Running and jumping through the fields!";
    let stemmed_text = stemmer.process_text(text);
    println!("Stemmed text results: {:?}\n", stemmed_text);

    //////////////////////
    // 3. single doc file
    //////////////////////
    print!("3. single doc file\n");
    let stemmer = PorterStemmer::new();

    // Process a document file
    let stemmed_words = stemmer.process_file_document("file_targets/test.txt")?;
    println!("-> Stemmed words: {:?}", stemmed_words);
    
    // Process with frequency counting
    let word_frequencies = stemmer.process_documentfile_with_frequencies("file_targets/test.txt")?;
    println!("-> Word frequencies: {:?}\n", word_frequencies);
    
    print!("3.2 noload single doc file\n");
    // Process document file with streaming
    let stems = stemmer.noload_process_filedocument_streaming(
        "file_targets/test.txt",
        "file_targets/test_stems.txt",
        1000, // chunk size
    )?;
    println!("-> Unique stems written to test_stems.txt");
    println!("{:?}", stems);
    
    // Process with frequency counting
    let frequencies = stemmer.noload_process_documentfile_frequencies_streaming(
        "file_targets/test.txt",
        "file_targets/test_stems.txt",
        "file_targets/test_frequencies.txt",
        1000, // chunk size
    )?;
    println!("-> Stem frequencies written to test_frequencies.txt\n");
    println!("{:?}", frequencies);
    
    ///////////////////
    // 4.1 csv corpus
    ///////////////////
    print!("4.1 csv corpus\n");
    // Process CSV and create document-term stem matrix
    stemmer.process_csv_to_bow_matrix(
        "file_targets/test_csv.csv", // read this input
        "output_with_bow.csv",       // path to output for results
        "stem_dictionary.txt",       // stem-dict output
        0,                           // assuming text is in column 1
    )?;
    
    
    ////////////////////////////////
    // 4.2 extra-noload csv corpus
    ////////////////////////////////
    print!("4.2 extra-noload csv corpus\n");
    // Process with explicit streaming and chunk size
    stemmer.noload_process_csvtobow_matrix_streaming(
        "file_targets/test_csv.csv",
        "noload_output_with_bow.csv",
        "noload_stem_dictionary.txt",
        0,  // text column
        1000, // process 1000 lines at a time
    )?;
    
    // Later, you can read the stem dictionary:
    let stems = PorterStemmer::read_stem_dictionary("stem_dictionary.txt")?;
    println!("Stems used in the BOW matrix:");
    for stem in stems {
        println!("{}", stem);
    }
    
    
    Ok(())
    
}

/*
TODO

BOW Sparse Porter Matrix

1. results output as csv (or some other recommended format)
2. function to process a .csv as input and output the same
csv with the "document-term stem matrix" added as one-hot fields.

This should not attempt to load the entire csv, 
rather read one line at a time, and append one line at a time
to the results.

*/