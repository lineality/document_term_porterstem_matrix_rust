//! # Document-Term Porter-Stem Matrix in Rust
//! document_term_porterstem_matrix_rust
//!
//!
//! Two Sweep Workflow:
//! 1. start an emtpy tokenizer lookup dict
//! 2. start 1st sweep: collect stem-tokens
//! 3. (1st sweep) load a row(or chunk of rows)
//! 4. (1st sweep) process one row
//! 5. (1st sweep) add each stem-token from each row to the lookup dict
//! 6. (1st sweep) at end of 1st sweep through all rows, save tokenizer dict as file (json)/(jsonl?)
//! 7. start 2nd sweep: add csv rows with counts
//! 8. (2nd sweep) iterate though all rows again
//! 9. (2nd sweep) add all token collumns to each row
//! 10. (2nd sweep) add value count for each token colum to each row
//! 11. save resulting .csv all old and new values
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
//! # Read stemsensure_output_dira
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

// use std::fmt::Write;
// use std::io::Write;

use std::f64::EPSILON;
use std::fs::{self, File};
use std::io::{
    self,
    BufRead,
    BufReader,
    Write,
};
use std::sync::Arc;
use std::collections::{
    HashMap,
    HashSet,
};
use std::sync::Mutex;
// use std::slice::range;
use std::path::Path;
use std::error::Error;
use std::fmt;
use std::fmt::Write as FmtWrite; // Note the alias to avoid confl

// 3rd Party Ug
use rayon::prelude::*;
use serde::{
    Serialize, 
    Deserialize,
};
use ndarray::{
    Array1,
    Array2, 
    ArrayView1,
    ArrayView2,
    arr2,

};

const NLTK_STOPWORDS: [&str; 127] = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    ];


#[derive(Debug)]
pub enum LogisticRegressionError {
    ConvergenceError(String),
    DimensionMismatch(String),
    InvalidInput(String),
}

impl fmt::Display for LogisticRegressionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LogisticRegressionError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
            LogisticRegressionError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            LogisticRegressionError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl Error for LogisticRegressionError {}

/// Represents feature importance derived from logistic regression
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    pub feature_index: usize,
    pub token: String,
    pub coefficient: f64,
    pub abs_importance: f64,
}

/// Logistic Regression model for binary classification
pub struct LogisticRegression {
    coefficients: Array1<f64>,
    intercept: f64,
    max_iterations: usize,
    learning_rate: f64,
    tolerance: f64,
}

impl LogisticRegression {
    /// Create a new LogisticRegression instance with default parameters
    pub fn new() -> Self {
        LogisticRegression {
            coefficients: Array1::zeros(0),
            intercept: 0.0,
            max_iterations: 1000,
            learning_rate: 0.1,  // Increased learning rate
            tolerance: 1e-4,
        }
    }

    /// Set maximum iterations for training
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set learning rate for gradient descent
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sigmoid function with numerical stability
    fn sigmoid(z: f64) -> f64 {
        if z > 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            exp_z / (1.0 + exp_z)
        }
    }

    // /// Fit the model to training data
    // pub fn fit(
    //     &mut self,
    //     x: ArrayView2<f64>,
    //     y: ArrayView1<f64>,
    // ) -> Result<(), LogisticRegressionError> {
    //     // Validate input dimensions
    //     if x.nrows() != y.len() {
    //         return Err(LogisticRegressionError::DimensionMismatch(
    //             "Number of samples in X and y must match".to_string(),
    //         ));
    //     }

    //     let n_samples = x.nrows();
    //     let n_features = x.ncols();

    //     // Initialize coefficients with explicit f64 type
    //     self.coefficients = Array1::<f64>::zeros(n_features);
    //     self.intercept = 0.0;

    //     // Gradient descent
    //     for iteration in 0..self.max_iterations {
    //         let mut gradient_coeffs: Array1<f64> = Array1::zeros(n_features);
    //         let mut gradient_intercept = 0.0;

    //         // Calculate predictions and gradients
    //         for i in 0..n_samples {
    //             let x_i = x.row(i);
    //             let y_i = y[i];

    //             let z = x_i.dot(&self.coefficients) + self.intercept;
    //             let prediction = Self::sigmoid(z);
    //             let error = prediction - y_i;

    //             // Update gradients
    //             for j in 0..n_features {
    //                 gradient_coeffs[j] += x_i[j] * error;
    //             }
    //             gradient_intercept += error;
    //         }

    //         // Update parameters
    //         let old_coefficients = self.coefficients.clone();
            
    //         // Update coefficients
    //         for j in 0..n_features {
    //             self.coefficients[j] -= self.learning_rate * gradient_coeffs[j] / n_samples as f64;
    //         }
            
    //         self.intercept -= self.learning_rate * gradient_intercept / n_samples as f64;

    //         // Check convergence
    //         let coeff_change = (&self.coefficients - &old_coefficients)
    //             .mapv(|x| x.abs())
    //             .sum();
    //         if coeff_change < self.tolerance {
    //             return Ok(());
    //         }
    //     }

    //     Err(LogisticRegressionError::ConvergenceError(
    //         "Failed to converge within maximum iterations".to_string(),
    //     ))
    // }

    /// Fit the model to training data
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(), LogisticRegressionError> {
        // Validate input dimensions
        if x.nrows() != y.len() {
            return Err(LogisticRegressionError::DimensionMismatch(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(LogisticRegressionError::InvalidInput(
                "Empty input data".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize coefficients with small random values
        self.coefficients = Array1::<f64>::zeros(n_features);
        self.intercept = 0.0;

        let mut prev_loss = f64::INFINITY;
        
        // Gradient descent
        for _iteration in 0..self.max_iterations {
            let mut gradient_coeffs: Array1<f64> = Array1::zeros(n_features);
            let mut gradient_intercept = 0.0;
            let mut current_loss = 0.0;

            // Calculate predictions and gradients
            for i in 0..n_samples {
                let x_i = x.row(i);
                let y_i = y[i];

                let z = x_i.dot(&self.coefficients) + self.intercept;
                let prediction = Self::sigmoid(z);
                
                // Calculate loss
                current_loss -= y_i * prediction.ln() + (1.0 - y_i) * (1.0 - prediction).ln();
                
                let error = prediction - y_i;

                // Update gradients
                for j in 0..n_features {
                    gradient_coeffs[j] += x_i[j] * error;
                }
                gradient_intercept += error;
            }

            current_loss /= n_samples as f64;

            // Early stopping if loss is increasing
            if current_loss > prev_loss * 1.5 {
                return Ok(());
            }
            prev_loss = current_loss;

            // Update parameters with normalized gradients
            for j in 0..n_features {
                self.coefficients[j] -= self.learning_rate * gradient_coeffs[j] / n_samples as f64;
            }
            self.intercept -= self.learning_rate * gradient_intercept / n_samples as f64;

            // Check convergence on loss instead of coefficients
            if (prev_loss - current_loss).abs() < self.tolerance {
                return Ok(());
            }
        }

        Ok(())  // Return Ok even if max iterations reached
    }
    /// Get feature importance scores
    pub fn get_feature_importance(&self) -> Vec<FeatureImportance> {
        let mut importance = Vec::with_capacity(self.coefficients.len());
        
        for (idx, &coef) in self.coefficients.iter().enumerate() {
            importance.push(FeatureImportance {
                feature_index: idx,
                token: format!("feature_{}", idx),
                coefficient: coef,
                abs_importance: coef.abs(),
            });
        }

        // Sort by absolute importance in descending order
        importance.sort_by(|a, b| b.abs_importance.partial_cmp(&a.abs_importance).unwrap());
        importance
    }
}

/// Calculate feature importance using logistic regression
pub fn calculate_logistic_regression_importance(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
) -> Result<Vec<FeatureImportance>, Box<dyn Error>> {
    // Convert labels to f64
    let y: Array1<f64> = Array1::from_iter(labels.iter().map(|&l| l as f64));

    // Create and train model
    let mut model = LogisticRegression::new()
        .with_max_iterations(1000)
        .with_learning_rate(0.01);

    model.fit(bow_matrix.view(), y.view())?;

    // Get feature importance
    Ok(model.get_feature_importance())
}

/// Print feature importance results
pub fn print_feature_importance(
    importance: &[FeatureImportance],
    stem_dictionary: Option<&[String]>,
    top_n: Option<usize>,
) {
    let n = top_n.unwrap_or(importance.len());
    println!("\nTop {} Features by Logistic Regression Importance:", n);
    println!("{:-<60}", "");

    for (i, feat) in importance.iter().take(n).enumerate() {
        let stem = stem_dictionary
            .and_then(|dict| dict.get(feat.feature_index))
            .map(|s| s.as_str())
            .unwrap_or("Unknown");

        println!("{}. Token: {} ({})", i + 1, feat.token, stem);
        println!("   Coefficient: {:.4}", feat.coefficient);
        println!("   Absolute Importance: {:.4}", feat.abs_importance);
        println!("{:-<60}", "");
    }
}

/// Represents mutual information scores for features
#[derive(Debug, Clone)]
pub struct MutualInformationScore {
    pub feature_index: usize,
    pub token: String,
    pub mi_score: f64,
    pub class_distributions: Vec<f64>,
}

/// Calculates mutual information between features and class labels
pub fn calculate_mutual_information(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
) -> Result<Vec<MutualInformationScore>, FeatureSelectionError> {  // Changed return type
    // Validate input dimensions
    if bow_matrix.nrows() != labels.len() {
        return Err(FeatureSelectionError::DimensionMismatch);
    }
    
    if bow_matrix.is_empty() || labels.is_empty() {
        return Err(FeatureSelectionError::EmptyInput);
    }

    // Calculate number of classes and features
    let num_features = bow_matrix.ncols();
    let num_samples = bow_matrix.nrows();
    let num_classes = labels.iter().max().unwrap_or(&0) + 1;

    let mut mi_scores = Vec::with_capacity(num_features);

    // Process each feature
    for feature_idx in 0..num_features {
        let feature_column = bow_matrix.column(feature_idx);
        
        // Calculate joint probability distribution P(X,Y)
        let joint_dist = calculate_joint_distribution(
            feature_column.view(),
            labels,
            num_classes,
            num_samples,
        );

        // Calculate marginal probabilities
        let class_marginals = calculate_class_marginals(&joint_dist);
        let feature_marginals = calculate_feature_marginals(&joint_dist);

        // Calculate mutual information
        let mi = calculate_mi_score(
            &joint_dist,
            &class_marginals,
            &feature_marginals,
            num_classes,
        );

        mi_scores.push(MutualInformationScore {
            feature_index: feature_idx,
            token: format!("feature_{}", feature_idx),
            mi_score: mi,
            class_distributions: class_marginals,
        });
    }

    // Sort by mutual information score in descending order
    mi_scores.sort_by(|a, b| b.mi_score.partial_cmp(&a.mi_score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(mi_scores)
}

/// Calculates joint probability distribution
fn calculate_joint_distribution(
    feature_column: ArrayView1<f64>,
    labels: &[usize],
    num_classes: usize,
    num_samples: usize,
) -> Array2<f64> {
    let mut joint_dist = Array2::zeros((num_classes, 2));

    for (idx, &feature_value) in feature_column.iter().enumerate() {
        let class = labels[idx];
        let bin = if feature_value > 0.0 { 1 } else { 0 };
        joint_dist[[class, bin]] += 1.0;
    }

    // Normalize to get probabilities
    joint_dist.mapv_inplace(|x| x / num_samples as f64);
    joint_dist
}

/// Calculates marginal probabilities for classes
fn calculate_class_marginals(joint_dist: &Array2<f64>) -> Vec<f64> {
    joint_dist.rows()
        .into_iter()
        .map(|row| row.sum())
        .collect()
}

/// Calculates marginal probabilities for features
fn calculate_feature_marginals(joint_dist: &Array2<f64>) -> Vec<f64> {
    joint_dist.columns()
        .into_iter()
        .map(|col| col.sum())
        .collect()
}

/// Calculates mutual information score
fn calculate_mi_score(
    joint_dist: &Array2<f64>,
    class_marginals: &[f64],
    feature_marginals: &[f64],
    num_classes: usize,
) -> f64 {
    let mut mi = 0.0;

    for c in 0..num_classes {
        for bin in 0..2 {
            let joint = joint_dist[[c, bin]];
            if joint > EPSILON {
                let expected = class_marginals[c] * feature_marginals[bin];
                if expected > EPSILON {
                    mi += joint * (joint / expected).ln();
                }
            }
        }
    }

    mi
}

/// Prints feature importance based on mutual information scores
pub fn print_mi_feature_importance(
    mi_scores: &[MutualInformationScore],
    stem_dictionary: Option<&[String]>,
    top_n: Option<usize>,
) {
    let n = top_n.unwrap_or(mi_scores.len());
    println!("\nTop {} Features by Mutual Information:", n);
    println!("{:-<60}", "");

    for (i, score) in mi_scores.iter().take(n).enumerate() {
        let stem = stem_dictionary
            .and_then(|dict| dict.get(score.feature_index))
            .map(|s| s.as_str())
            .unwrap_or("Unknown");

        println!("{}. Token: {} ({})", i + 1, score.token, stem);
        println!("   MI Score: {:.4}", score.mi_score);
        println!("   Class Distribution: {:?}", score.class_distributions);
        println!("{:-<60}", "");
    }
}


/// Ensures the directory for a given path exists
fn ensure_directory_for_path(path: &str) -> io::Result<()> {
    if let Some(dir) = Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}

fn ensure_directories() -> io::Result<()> {
    fs::create_dir_all("file_targets")?;
    fs::create_dir_all("output")?;
    Ok(())
}



/// Custom error type for feature selection operations
#[derive(Debug)]
pub enum FeatureSelectionError {
    EmptyInput,
    DimensionMismatch,
    InvalidLabels,
    ComputationError(String),
}

// Add this implementation after the FeatureSelectionError definition
impl From<FeatureSelectionError> for std::io::Error {
    fn from(error: FeatureSelectionError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, error.to_string())
    }
}

impl fmt::Display for FeatureSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FeatureSelectionError::EmptyInput => write!(f, "Empty input data"),
            FeatureSelectionError::DimensionMismatch => write!(f, "Dimension mismatch between features and labels"),
            FeatureSelectionError::InvalidLabels => write!(f, "Invalid labels detected"),
            FeatureSelectionError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl Error for FeatureSelectionError {}

// #[derive(Debug, Clone)]
// pub struct NormalizedCorrelation {
//     pub raw_score: f64,
//     pub normalized_score: f64,  // 0 to 1 scale
//     pub strength: CorrelationStrength,
// }

// Comprehensive Feature Analysis

/// Represents comprehensive feature analysis results
#[derive(Debug, Clone)]
pub struct FeatureAnalysis {
    pub feature_index: usize,
    pub token: String,
    pub chi_square_value: f64,
    pub mutual_info_score: f64,
    pub logistic_coef: f64,
    pub combined_score: f64,
}

/// Calculate Chi-Square scores for each feature
fn calculate_chi_square_scores(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
) -> Result<Vec<f64>, FeatureSelectionError> {
    let num_features = bow_matrix.ncols();
    let num_classes = labels.iter().max().unwrap_or(&0) + 1;
    let mut scores = Vec::with_capacity(num_features);

    for feature_idx in 0..num_features {
        let feature_column = bow_matrix.column(feature_idx);
        
        // Create contingency table
        let contingency = create_contingency_table(feature_column, labels, num_classes)?;
        
        // Calculate Chi-Square statistic
        let chi_square = calculate_chi_square(&contingency)?;
        scores.push(chi_square);
    }

    Ok(scores)
}

/// Calculate Mutual Information scores for each feature
fn calculate_mutual_information_scores(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
) -> Result<Vec<f64>, FeatureSelectionError> {
    let num_features = bow_matrix.ncols();
    let num_samples = bow_matrix.nrows();
    let num_classes = labels.iter().max().unwrap_or(&0) + 1;
    
    let mut scores = Vec::with_capacity(num_features);

    for feature_idx in 0..num_features {
        let feature_column = bow_matrix.column(feature_idx);
        
        // Calculate joint probability distribution
        let joint_dist = calculate_joint_distribution(
            feature_column,
            labels,
            num_classes,
            num_samples,
        );

        // Calculate marginal probabilities
        let class_marginals = calculate_class_marginals(&joint_dist);
        let feature_marginals = calculate_feature_marginals(&joint_dist);

        // Calculate mutual information
        let mi = calculate_mi_score(
            &joint_dist,
            &class_marginals,
            &feature_marginals,
            num_classes,
        );

        scores.push(mi);
    }

    Ok(scores)
}

/// Calculate Logistic Regression coefficients for feature importance
fn calculate_logistic_regression_scores(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
) -> Result<Vec<f64>, FeatureSelectionError> {
    let num_features = bow_matrix.ncols();
    let mut scores = Vec::with_capacity(num_features);

    // Convert labels to f64
    let y: Array1<f64> = Array1::from_iter(labels.iter().map(|&l| l as f64));

    // Create and train logistic regression model
    let mut model = LogisticRegression::new()
        .with_max_iterations(1000)
        .with_learning_rate(0.01);

    match model.fit(bow_matrix.view(), y.view()) {
        Ok(_) => {
            // Get absolute values of coefficients as importance scores
            for coef in model.coefficients.iter() {
                scores.push(coef.abs());
            }
            Ok(scores)
        },
        Err(_) => Err(FeatureSelectionError::ComputationError(
            "Failed to fit logistic regression model".to_string()
        )),
    }
}

/// Performs comprehensive feature analysis using multiple methods
pub fn analyze_feature_correlations(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
    stem_dictionary: Option<&[String]>,
) -> Result<Vec<FeatureAnalysis>, FeatureSelectionError> {
    // Validate input
    if bow_matrix.is_empty() || labels.is_empty() {
        return Err(FeatureSelectionError::EmptyInput);
    }
    if bow_matrix.nrows() != labels.len() {
        return Err(FeatureSelectionError::DimensionMismatch);
    }

    let num_features = bow_matrix.ncols();
    let mut results = Vec::with_capacity(num_features);

    // Calculate feature importance using different methods
    let chi_square_scores = calculate_chi_square_scores(bow_matrix, labels)?;
    let mutual_info_scores = calculate_mutual_information_scores(bow_matrix, labels)?;
    let logistic_scores = calculate_logistic_regression_scores(bow_matrix, labels)?;

    // Normalize scores explicit f64 versions:
    let chi_square_max = chi_square_scores.iter()
        .fold(0.0_f64, |a: f64, &b| f64::max(a, b));
    let mutual_info_max = mutual_info_scores.iter()
        .fold(0.0_f64, |a: f64, &b| f64::max(a, b));
    let logistic_max = logistic_scores.iter()
        .fold(0.0_f64, |a: f64, &b| f64::max(a, b));
    
    // Combine results
    for i in 0..num_features {
        let token = match stem_dictionary {
            Some(dict) => dict.get(i)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("feature_{}", i)),
            None => format!("feature_{}", i),
        };

        let chi_square_norm = if chi_square_max > 0.0 { 
            chi_square_scores[i] / chi_square_max 
        } else { 
            0.0 
        };
        let mutual_info_norm = if mutual_info_max > 0.0 { 
            mutual_info_scores[i] / mutual_info_max 
        } else { 
            0.0 
        };
        let logistic_norm = if logistic_max > 0.0 { 
            logistic_scores[i] / logistic_max 
        } else { 
            0.0 
        };

        // Calculate combined score (weighted average)
        let combined_score = (chi_square_norm + mutual_info_norm + logistic_norm) / 3.0;

        results.push(FeatureAnalysis {
            feature_index: i,
            token,
            chi_square_value: chi_square_scores[i],
            mutual_info_score: mutual_info_scores[i],
            logistic_coef: logistic_scores[i],
            combined_score,
        });
    }

    // Sort by combined score in descending order
    results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

    Ok(results)
}


/// Save analysis results to a file in CSV format
pub fn save_feature_analysis_csv(
    analysis: &[FeatureAnalysis],
    output_path: &str,
) -> io::Result<()> {
    let mut file = File::create(output_path)?;
    
    // Write header
    writeln!(file, "Token,Chi_Square,Mutual_Information,Logistic_Coefficient,Combined_Score")?;

    // Write data
    for result in analysis {
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6}",
            result.token,
            result.chi_square_value,
            result.mutual_info_score,
            result.logistic_coef,
            result.combined_score
        )?;
    }

    Ok(())
}

/// Generate a detailed summary report
pub fn generate_analysis_summary(
    analysis: &[FeatureAnalysis],
    output_path: Option<&str>,
) -> io::Result<String> {
    let mut summary = String::new();
    
    // Calculate basic statistics
    let total_features = analysis.len();
    let avg_chi_square: f64 = analysis.iter().map(|x| x.chi_square_value).sum::<f64>() / total_features as f64;
    let avg_mi: f64 = analysis.iter().map(|x| x.mutual_info_score).sum::<f64>() / total_features as f64;
    let avg_coef: f64 = analysis.iter().map(|x| x.logistic_coef).sum::<f64>() / total_features as f64;

    // Format summary using write! instead of writeln!
    let _ = write!(summary, "Feature Analysis Summary\n");
    let _ = write!(summary, "{:-<50}\n", "");
    let _ = write!(summary, "Total Features Analyzed: {}\n", total_features);
    let _ = write!(summary, "Average Chi-Square Value: {:.4}\n", avg_chi_square);
    let _ = write!(summary, "Average Mutual Information: {:.4}\n", avg_mi);
    let _ = write!(summary, "Average Logistic Coefficient: {:.4}\n", avg_coef);
    let _ = write!(summary, "\nTop 10 Most Important Features:\n");
    
    for (i, result) in analysis.iter().take(10).enumerate() {
        let _ = write!(
            summary,
            "{}. {} (Score: {:.4})\n", 
            i + 1, 
            result.token, 
            result.combined_score
        );
    }

    // Save to file if path provided
    if let Some(path) = output_path {
        fs::write(path, &summary)?;
    }

    Ok(summary)
}

/// Example usage function
pub fn analyze_and_report(
    analysis_results: Vec<FeatureAnalysis>,
    output_dir: &str,
) -> io::Result<()> {
    // Ensure output directory exists
    fs::create_dir_all(output_dir)?;

    // Print detailed analysis
    print_feature_analysis(&analysis_results, Some(20));

    // Save to CSV
    let csv_path = format!("{}/feature_analysis.csv", output_dir);
    save_feature_analysis_csv(&analysis_results, &csv_path)?;

    // Generate and save summary
    let summary_path = format!("{}/analysis_summary.txt", output_dir);
    let summary = generate_analysis_summary(&analysis_results, Some(&summary_path))?;
    println!("\nAnalysis Summary:\n{}", summary);

    Ok(())
}

/// Print feature analysis results with customizable formatting
pub fn print_feature_analysis(
    analysis: &[FeatureAnalysis],
    top_n: Option<usize>,
) {
    let n = top_n.unwrap_or(analysis.len());
    println!("\nTop {} Features by Combined Analysis:", n);
    println!("{:-<80}", "");

    for (i, result) in analysis.iter().take(n).enumerate() {
        println!("{}. Token: {}", i + 1, result.token);
        println!("   Chi-Square Value: {:.4}", result.chi_square_value);
        println!("   Mutual Information: {:.4}", result.mutual_info_score);
        println!("   Logistic Coefficient: {:.4}", result.logistic_coef);
        println!("   Combined Score: {:.4}", result.combined_score);
        println!("{:-<80}", "");
    }
}




/// Represents the correlation statistics for a single feature
#[derive(Debug, Clone)]
pub struct FeatureCorrelation {
    pub token: String,
    pub chi_square_value: f64,
    pub p_value: f64,
    pub degrees_of_freedom: usize,
}

/// Creates a contingency table for chi-square calculation
fn create_contingency_table(
    feature_column: ArrayView1<f64>,
    labels: &[usize],
    num_classes: usize,
) -> Result<Array2<f64>, FeatureSelectionError> {
    // Initialize 2x2 contingency table for each class
    let mut table = Array2::zeros((2, num_classes));
    
    for (feat_val, &label) in feature_column.iter().zip(labels) {
        if label >= num_classes {
            return Err(FeatureSelectionError::InvalidLabels);
        }
        
        let row = if *feat_val > 0.0 { 1 } else { 0 };
        table[[row, label]] += 1.0;
    }
    
    Ok(table)
}

/// Calculates chi-square statistic from contingency table
fn calculate_chi_square(table: &Array2<f64>) -> Result<f64, FeatureSelectionError> {
    let total: f64 = table.sum();
    if total == 0.0 {
        return Err(FeatureSelectionError::ComputationError(
            "Empty contingency table".to_string()
        ));
    }

    let row_sums = table.sum_axis(ndarray::Axis(1));
    let col_sums = table.sum_axis(ndarray::Axis(0));
    
    let mut chi_square = 0.0;
    
    for i in 0..table.nrows() {
        for j in 0..table.ncols() {
            let observed = table[[i, j]];
            let expected = (row_sums[i] * col_sums[j]) / total;
            
            if expected > 0.0 {
                chi_square += (observed - expected).powi(2) / expected;
            }
        }
    }
    
    Ok(chi_square)
}

/// Calculates p-value from chi-square statistic and degrees of freedom
fn calculate_p_value(chi_square: f64, _df: usize) -> f64 {
    // Simple p-value approximation
    // Added underscore to df to silence unused variable warning
    (-0.5 * chi_square).exp()
}

/// Performs chi-square feature selection on the given BOW matrix
pub fn chi_square_feature_selection(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
    significance_threshold: f64,
) -> Result<Vec<FeatureCorrelation>, FeatureSelectionError> {
    // Input validation
    if bow_matrix.is_empty() || labels.is_empty() {
        return Err(FeatureSelectionError::EmptyInput);
    }
    
    if bow_matrix.nrows() != labels.len() {
        return Err(FeatureSelectionError::DimensionMismatch);
    }
    
    // Find number of unique classes
    let num_classes = labels.iter()
        .max()
        .map(|&max| max + 1)
        .ok_or(FeatureSelectionError::InvalidLabels)?;
    
    let mut correlations = Vec::new();
    
    // Process each feature
    for feature_idx in 0..bow_matrix.ncols() {
        let feature_column = bow_matrix.column(feature_idx);
        
        // Create and analyze contingency table
        let contingency = create_contingency_table(feature_column, labels, num_classes)?;
        let chi_square = calculate_chi_square(&contingency)?;
        
        // Calculate degrees of freedom
        let df = (contingency.nrows() - 1) * (contingency.ncols() - 1);
        let p_value = calculate_p_value(chi_square, df);
        
        correlations.push(FeatureCorrelation {
            token: format!("feature_{}", feature_idx),
            chi_square_value: chi_square,
            p_value,
            degrees_of_freedom: df,
        });
    }
    
    // Sort by chi-square value in descending order
    correlations.sort_by(|a, b| b.chi_square_value.partial_cmp(&a.chi_square_value)
        .unwrap_or(std::cmp::Ordering::Equal));
    
    // Filter by significance threshold
    Ok(correlations.into_iter()
        .filter(|c| c.p_value <= significance_threshold)
        .collect())
}

/// Helper function to print feature correlation results
pub fn print_feature_correlations(
    correlations: &[FeatureCorrelation],
    top_n: Option<usize>,
    stem_dictionary: Option<&[String]>,
) {
    let correlations_to_show = top_n.unwrap_or(correlations.len());
    
    println!("\nTop {} Feature Correlations:", correlations_to_show);
    println!("{:-<60}", "");
    
    for (i, correlation) in correlations.iter().take(correlations_to_show).enumerate() {
        let token_index = correlation.token
            .split('_')
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok());
        
        let stem = token_index
            .and_then(|idx| stem_dictionary.and_then(|dict| dict.get(idx)))
            .map(|s| s.as_str())
            .unwrap_or("Unknown");
        
        println!("{}. Token: {} ({})", i + 1, correlation.token, stem);
        println!("   Chi-Square Value: {:.4}", correlation.chi_square_value);
        println!("   P-Value: {:.4}", correlation.p_value);
        println!("   Degrees of Freedom: {}", correlation.degrees_of_freedom);
        println!("{:-<60}", "");
    }
}

#[derive(Debug)]
struct MultiFileProcessor {
    input_files_list: Vec<String>,
    output_dir: String,
    text_column: usize,
}

impl MultiFileProcessor {
    /// Creates a new MultiFileProcessor instance
    /// 
    /// # Arguments
    /// * `input_files_list` - List of paths to input CSV files to process
    /// * `output_dir` - Directory where output files will be written
    /// * `text_column` - Zero-based index specifying which column in the CSV contains the text to process
    fn new(
        input_files_list: Vec<String>,
        output_dir: String,
        text_column: usize,
    ) -> Self {
        Self {
            input_files_list,
            output_dir,
            text_column,
        }
    }

    fn process_all_files(&self) -> io::Result<()> {
        // Ensure output directory exists
        fs::create_dir_all(&self.output_dir)?;

        // Phase 1: Collect stems across all files
        let combined_tokenizer_dict = self.collect_stems_from_all_files()?;
        
        // Save the combined dictionary
        let dict_path = format!("{}/combined_tokenizer_dict.json", self.output_dir);
        combined_tokenizer_dict.save_to_json(&dict_path)?;

        // Phase 2: Create BOW matrices for each file
        self.create_bow_matrices_for_all_files(&combined_tokenizer_dict)?;

        Ok(())
    }

    fn collect_stems_from_all_files(&self) -> io::Result<TokenizerDict> {
        let mut combined_dict = TokenizerDict::new();

        // Process each file
        for input_file in &self.input_files_list {
            println!("Collecting stems from: {}", input_file);
            combined_dict.first_sweep(input_file, self.text_column)?;
        }

        Ok(combined_dict)
    }

    fn create_bow_matrices_for_all_files(&self, combined_tokenizer_dict: &TokenizerDict) -> io::Result<()> {
        for input_file in &self.input_files_list {
            let file_name = Path::new(input_file)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
                
            let output_path = format!("{}/bow_matrix_{}.csv", self.output_dir, file_name);
            
            println!("Creating BOW matrix for: {}", input_file);
            combined_tokenizer_dict.second_sweep(
                input_file, 
                &output_path, 
                self.text_column,
            )?;
        }

        Ok(())
    }
}


// #[derive(Debug)]
// struct TokenizerConfig {
//     remove_stopwords: bool,
//     text_column: usize,
//     // Add other configuration options as needed
// }

#[derive(Debug, Serialize, Deserialize)]
struct TokenizerDict {
    stems: HashMap<String, usize>,
    total_docs: usize,
}

impl TokenizerDict {
    /// First sweep through CSV to collect stems
    /// 
    /// # Arguments
    /// * `single_file_input_path` - Path to input CSV file
    /// * `text_column` - Zero-based index of column containing text to process
    fn new() -> Self {
        TokenizerDict {
            stems: HashMap::new(),
            total_docs: 0,
        }
    }

    // fn process_with_config(&mut self, single_file_input_path: &str, config: &TokenizerConfig) -> io::Result<()> {
    //     // Regular first sweep
    //     self.first_sweep(single_file_input_path, config.text_column)?;
        
    //     // Apply stopword filtering if configured
    //     if config.remove_stopwords {
    //         self.filter_stop_words();
    //     }
        
    //     Ok(())
    // }    
    
    /// Calculate term frequency for a document
    fn calculate_tf(&self, term_counts: &HashMap<String, usize>, total_terms: usize) -> HashMap<String, f64> {
        let mut tf = HashMap::new();
        for (term, &count) in term_counts {
            tf.insert(term.clone(), (count as f64) / (total_terms as f64));
        }
        tf
    }

    /// Calculate inverse document frequency
    fn calculate_idf(&self, term: &str, total_docs: usize) -> f64 {
        let docs_with_term = self.stems.get(term).unwrap_or(&0);
        (total_docs as f64 / (1.0 + *docs_with_term as f64)).ln()
    }

    /// Calculate TF-IDF for a document
    pub fn calculate_tfidf_for_document(
        &self,
        term_counts: &HashMap<String, usize>,
        total_terms: usize,
    ) -> HashMap<String, f64> {
        let tf = self.calculate_tf(term_counts, total_terms);
        let mut tfidf = HashMap::new();
        
        for (term, tf_value) in tf {
            let idf = self.calculate_idf(&term, self.total_docs);
            tfidf.insert(term, tf_value * idf);
        }
        
        tfidf
    }

    /// Modify second_sweep to include TF-IDF scores
    /// Second sweep with TF-IDF scoring
    /// 
    /// # Arguments
    /// * `single_file_input_path` - Path to input CSV file
    /// * `output_path` - Path where TF-IDF matrix CSV will be written
    /// * `text_column` - Zero-based index of column containing text to process
    pub fn second_sweep_with_tfidf(
        &self,
        single_file_input_path: &str,
        output_path: &str,
        text_column: usize,
    ) -> io::Result<()> {
        let input_file = File::open(single_file_input_path)?;
        let mut output_file = File::create(output_path)?;
        let reader = BufReader::new(input_file);
        let mut stemmer = PorterStemmer::new();

        // Write header with TF-IDF columns
        let mut lines = reader.lines();
        if let Some(header_line) = lines.next() {
            let original_header = header_line?;
            let stem_header: String = self.stems.keys()
                .flat_map(|stem| {
                    vec![
                        format!("stem_{}_freq", stem),
                        format!("stem_{}_tfidf", stem),
                    ]
                })
                .collect::<Vec<_>>()
                .join(",");
            writeln!(output_file, "{},{}", original_header, stem_header)?;
        }

        // Process documents
        for line_result in lines {
            let line = line_result?;
            let fields: Vec<&str> = line.split(',').collect();
            
            // Count stems and calculate total terms
            let mut term_counts = HashMap::new();
            let mut total_terms = 0;
            
            if let Some(text) = fields.get(text_column) {
                // println!("Processing text: {}", text);  // Debug log                
                let words = PorterStemmer::extract_words(text);
                total_terms = words.len();
                
                for word in words {
                    let stem = stemmer.stem(&word);
                    *term_counts.entry(stem).or_insert(0) += 1;
                }
            } else {
                println!("Warning: No text found at column {}", text_column);
                continue;
            }

            // Calculate TF-IDF scores
            let tfidf_scores = self.calculate_tfidf_for_document(&term_counts, total_terms);

            // Write original line plus frequencies and TF-IDF scores
            let scores: String = self.stems.keys()
                .flat_map(|stem| {
                    let freq = term_counts.get(stem).cloned().unwrap_or(0);
                    let tfidf = tfidf_scores.get(stem).cloned().unwrap_or(0.0);
                    vec![
                        freq.to_string(),
                        format!("{:.4}", tfidf),
                    ]
                })
                .collect::<Vec<_>>()
                .join(",");
            
            writeln!(output_file, "{},{}", line, scores)?;
        }

        Ok(())
    }
    
    // Add stop word filtering
    fn filter_stop_words(&mut self) {
        // Convert NLTK_STOPWORDS array to HashSet for efficient lookup
        let stop_words: HashSet<&str> = NLTK_STOPWORDS.iter().copied().collect();
        
        // Filter out stems that match stopwords
        self.stems.retain(|stem, _| !stop_words.contains(stem.as_str()));
    }


    /// First sweep with stopword filtering
    /// 
    /// # Arguments
    /// * `single_file_input_path` - Path to input CSV file
    /// * `text_column` - Zero-based index of column containing text to process
    fn first_sweep_with_stopwords(&mut self, single_file_input_path: &str, text_column: usize) -> io::Result<()> {
        // Perform regular first sweep
        self.first_sweep(single_file_input_path, text_column)?;
        
        // Filter out stopwords
        self.filter_stop_words();
        
        Ok(())
    }

    /// Save tokenizer dictionary to JSON
    fn save_to_json(&self, path: &str) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load tokenizer dictionary from JSON
    fn load_from_json(path: &str) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let dict: TokenizerDict = serde_json::from_reader(reader)?;
        Ok(dict)
    }
    
    // First sweep: Process each row individually
    fn first_sweep(&mut self, single_file_input_path: &str, text_column: usize) -> io::Result<()> {
        let file = File::open(single_file_input_path)?;
        let reader = BufReader::new(file);
        let mut stemmer = PorterStemmer::new();

        // Skip header
        let mut lines = reader.lines();
        let _ = lines.next();

        // Process each row individually
        for line_result in lines {
            let line = line_result?;
            let fields: Vec<&str> = line.split(',').collect();
            
            if let Some(text) = fields.get(text_column) {
                // Extract and count stems for this document
                // println!("Processing text: {}", text);  // Debug log                
                let words = PorterStemmer::extract_words(text);
                for word in words {
                    let stem = stemmer.stem(&word);
                    *self.stems.entry(stem).or_insert(0) += 1;
                }
            } else {
                println!("Warning: No text found at column {}", text_column);
                continue;
            }
            
            self.total_docs += 1;
        }

        Ok(())
    }

    // Second sweep: Create BOW matrix using collected stems
    /// Second sweep through CSV to create BOW (Bag of Words) matrix
    /// 
    /// # Arguments
    /// * `single_file_input_path` - Path to input CSV file
    /// * `output_path` - Path where BOW matrix CSV will be written
    /// * `text_column` - Zero-based index of column containing text to process
    fn second_sweep(
        &self, 
        single_file_input_path: &str, 
        output_path: &str, 
        text_column: usize
    ) -> io::Result<()> {
        let input_file = File::open(single_file_input_path)?;
        let mut output_file = File::create(output_path)?;
        let reader = BufReader::new(input_file);
        let mut stemmer = PorterStemmer::new();

        // Process header
        let mut lines = reader.lines();
        if let Some(header_line) = lines.next() {
            let original_header = header_line?;
            // Combine original header with stem columns
            let stem_header: String = self.stems.keys()
                .map(|stem| format!("stem_{}_freq", stem))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(output_file, "{},{}", original_header, stem_header)?;
        }

        // Process each document
        for line_result in lines {
            let line = line_result?;
            let fields: Vec<&str> = line.split(',').collect();
            
            // Count stems in this document
            let mut doc_stem_freq = HashMap::new();
            if let Some(text) = fields.get(text_column) {
                // println!("Processing text: {}", text);  // Debug log                
                let words = PorterStemmer::extract_words(text);
                for word in words {
                    let stem = stemmer.stem(&word);
                    *doc_stem_freq.entry(stem).or_insert(0) += 1;
                }
            } else {
                println!("Warning: No text found at column {}", text_column);
                continue;
            }

            // Write original line plus frequencies for all known stems
            let freq_values: String = self.stems.keys()
                .map(|stem| doc_stem_freq.get(stem).cloned().unwrap_or(0).to_string())
                .collect::<Vec<_>>()
                .join(",");
            
            writeln!(output_file, "{},{}", line, freq_values)?;
        }

        Ok(())
    }
    
} // end of impl TokenizerDict

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
        dict_path: &str,  // Full path
    ) -> io::Result<()> {
        ensure_directory_for_path(dict_path)?;
        let mut dict_file = File::create(dict_path)?;

        let mut sorted_stems: Vec<_> = stems.iter().collect();
        sorted_stems.sort(); // Sort for consistent output
        
        println!("Writing {} stems to dictionary: {}", stems.len(), dict_path);
        for stem in sorted_stems {
            writeln!(dict_file, "{}", stem)?;
            println!("Wrote stem: {}", stem); // Debug print
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
    /// Process CSV in a streaming fashion to create BOW matrix
    /// 
    /// # Arguments
    /// * `single_file_input_path` - Path to input CSV file
    /// * `output_path` - Path for output BOW matrix CSV
    /// * `dict_path` - Path for output stem dictionary
    /// * `text_column` - Zero-based index of column containing text to process
    /// * `chunk_size` - Number of rows to process at once
    pub fn noload_process_csvtobow_matrix_streaming(
        &self,
        single_file_input_path: &str,
        output_path: &str, // Full path
        dict_path: &str,   // Full path
        text_column: usize,
        chunk_size: usize,
    ) -> io::Result<()> {
        // Ensure output directories exist
        ensure_directory_for_path(output_path)?;
        ensure_directory_for_path(dict_path)?;
        
        // First pass: collect stems using streaming
        let unique_stems = self.collect_unique_stems_streaming(single_file_input_path, text_column)?;
        println!("Collected stems: {:?}", unique_stems);  // Debug print
        
        // Write stems dictionary
        self.write_stem_dictionary(&unique_stems, &dict_path)?;
        
        // Convert to ordered Vec
        let stem_vec: Vec<String> = unique_stems.into_iter().collect();
        
        // Second pass: create BOW matrix streaming
        let input = BufReader::new(File::open(single_file_input_path)?);
        let mut output = File::create(output_path)?;
        
        // Write header - preserve original header and add stem columns
        let mut lines = input.lines();
        if let Some(header_line) = lines.next() {
            let original_header = header_line?;
            let stem_header: String = stem_vec
                .iter()
                .map(|stem| format!("stem_{}", stem))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(output, "{},{}", original_header, stem_header)?;
        }

        // Process remaining lines in chunks
        let mut buffer = Vec::with_capacity(chunk_size);
        
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
        single_file_input_path: &str,
        text_column: usize,
    ) -> io::Result<HashSet<String>> {
        let file = File::open(single_file_input_path)?;
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

    
    /// Process CSV and create both stem dictionary and BOW matrix
    /// 
    /// # Arguments
    /// * `single_file_input_path` - Path to input CSV file
    /// * `output_path` - Path for output BOW matrix CSV
    /// * `dict_path` - Path for output stem dictionary
    /// * `text_column` - Zero-based index of column containing text to process
    /// 
    /// # Returns
    /// Result indicating success or failure
    /// Process CSV and create both stem dictionary and BOW matrix
    pub fn process_csv_to_bow_matrix(
        &self,
        single_file_input_path: &str,
        output_path: &str,
        dict_path: &str,
        text_column: usize,
    ) -> io::Result<()> {
        // Ensure output directories exist
        ensure_directory_for_path(output_path)?;
        ensure_directory_for_path(dict_path)?;
        
        // First pass: collect unique stems
        let unique_stems = self.collect_unique_stems(single_file_input_path, text_column)?;
        
        // Write stem dictionary
        self.write_stem_dictionary(&unique_stems, dict_path)?;
        
        // Convert HashSet to Vec for ordered access
        let stem_vec: Vec<String> = unique_stems.into_iter().collect();
        
        // Second pass: create the document-term matrix
        let input_file = File::open(single_file_input_path)?;
        let output_file = File::create(output_path)?;
        self.create_bow_matrix(
            input_file,
            output_file,
            text_column,
            &stem_vec,
        )?;
        
        Ok(())
    }

    /// Collect unique stems from CSV file
    /// 
    /// # Arguments
    /// * `single_file_input_path` - Path to input CSV file
    /// * `text_column` - Zero-based index of column containing text to process
    fn collect_unique_stems(
        &self,
        single_file_input_path: &str,
        text_column: usize,
    ) -> io::Result<HashSet<String>> {
        let file = File::open(single_file_input_path)?;
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
    /// * `single_file_input_path` - Path to input text file
    /// * `dict_path` - Path for output stem dictionary
    /// * `chunk_size` - Number of lines to process at once
    /// 
    /// # Returns
    /// Result containing HashSet of stems for immediate use if needed
    pub fn noload_process_filedocument_streaming(
        &self,
        single_file_input_path: &str,
        output_dict_path: &str,
        chunk_size: usize,
    ) -> io::Result<HashSet<String>> {
        
        ensure_directory_for_path(output_dict_path)?;
        
        let file = File::open(single_file_input_path)?;
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
        self.write_stem_dictionary(&unique_stems, &output_dict_path)?;

        Ok(unique_stems)
    }

    /// Process document with frequency counting, streaming style
    pub fn noload_process_documentfile_frequencies_streaming(
        &self,
        single_file_input_path: &str,
        output_dict_path: &str,
        output_freq_path: &str,
        chunk_size: usize,
    ) -> io::Result<HashMap<String, usize>> {
        
        ensure_directory_for_path(single_file_input_path)?;
        ensure_directory_for_path(output_dict_path)?;
        ensure_directory_for_path(output_freq_path)?;
        
        let file = File::open(single_file_input_path)?;
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
        self.write_stem_dictionary(&unique_stems, &output_dict_path)?;

        // Write frequencies to separate file
        let mut freq_file = File::create(output_freq_path)?;
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

    // /// Returns true if the word ends with the given string
    // fn ends_with(&mut self, s: &str) -> bool {
    //     let length = s.len();
    //     if length > self.k - self.k0 + 1 { return false; }
        
    //     let end = &self.buffer[(self.k + 1 - length)..=self.k];
    //     let s_chars: Vec<char> = s.chars().collect();
        
    //     if end != &s_chars[..] { return false; }
        
    //     self.j = self.k - length;  // thread 'main' panicked at src/main.rs:2618:18: ->attempt to subtract with overflow
    //     true
    // }

    /// Returns true if the word ends with the given string
    /// 
    /// # Arguments
    /// * `s` - The suffix string to check for
    /// 
    /// # Returns
    /// * `bool` - True if the word ends with the given string
    /// 
    /// # Safety
    /// Performs bounds checking to prevent integer overflow
    fn ends_with(&mut self, s: &str) -> bool {
        let length = s.len();
        
        // // Debug logging
        // println!("Checking ends_with: word={:?}, suffix={}, k={}, k0={}, length={}", 
        //     self.buffer, s, self.k, self.k0, length);
        
        // Early returns with additional safety checks
        if length == 0 { return true; }
        if self.k < length - 1 { return false; }  // Not enough characters
        if length > self.k - self.k0 + 1 { return false; }
        
        // Safe subtraction with checked arithmetic
        let start_pos = match self.k.checked_sub(length - 1) {
            Some(pos) => pos,
            None => {
                println!("Warning: Arithmetic overflow prevented in ends_with");
                return false;
            }
        };
        
        // Get the end slice safely
        let end = &self.buffer[start_pos..=self.k];
        let s_chars: Vec<char> = s.chars().collect();
        
        if end != &s_chars[..] { return false; }
        
        // Safe subtraction for setting j
        self.j = match self.k.checked_sub(length) {
            Some(pos) => pos,
            None => {
                println!("Warning: Arithmetic overflow prevented when setting j");
                return false;
            }
        };
        
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
        // println!("Attempting to stem word: {}", word);  // Debug print
    
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


/// Represents a feature's correlation analysis with both raw and normalized scores
#[derive(Debug, Clone)]
pub struct NormalizedFeatureAnalysis {
    /// Token or byte value being analyzed
    pub token: String,
    
    /// Index of feature in the original feature matrix
    pub feature_index: usize,
    
    // Raw (unnormalized) correlation scores
    /// Chi-square test statistic (raw)
    pub raw_chi_square: f64,
    /// Mutual information score (raw)
    pub raw_mutual_info: f64,
    /// Logistic regression coefficient (raw)
    pub raw_logistic_coef: f64,
    
    // Normalized scores (0 to 1 scale)
    /// Normalized chi-square score
    pub norm_chi_square: f64,
    /// Normalized mutual information score
    pub norm_mutual_info: f64,
    /// Normalized logistic regression coefficient
    pub norm_logistic_coef: f64,
    
    // Combined metrics
    /// Combined weighted correlation score (average of normalized scores)
    pub weighted_correlation: f64,
    /// Direction of correlation: 1 (positive), -1 (negative), 0 (neutral)
    pub correlation_direction: i8,
}

impl NormalizedFeatureAnalysis {
    /// Normalizes a vector of scores to the range [0, 1]
    /// 
    /// # Arguments
    /// * `scores` - Vector of raw scores to normalize
    /// 
    /// # Returns
    /// * Vector of normalized scores in range [0, 1]
    /// 
    /// # Notes
    /// Uses min-max normalization: (x - min) / (max - min)
    /// Returns zeros if all scores are identical
    fn normalize_scores(scores: &[f64]) -> Vec<f64> {
        if scores.is_empty() {
            return Vec::new();
        }
        
        let min = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Add small epsilon to prevent division by zero
        let epsilon = f64::EPSILON * 100.0;  // Slightly larger epsilon for numerical stability
        let range = (max - min) + epsilon;
        
        scores.iter()
            .map(|&x| {
                if range < epsilon {
                    0.0
                } else {
                    ((x - min) / range).max(0.0).min(1.0)
                }
            })
            .collect()
    }
        
    
    
    // fn normalize_scores(scores: &[f64]) -> Vec<f64> {
    //     if scores.is_empty() {
    //         return Vec::new();
    //     }
        
    //     // Find min and max values
    //     let min = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    //     let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
    //     // Check for division by zero case
    //     if (max - min).abs() < f64::EPSILON {
    //         return vec![0.0; scores.len()];
    //     }
        
    //     // // Perform min-max normalization
    //     // scores.iter()
    //     //     .map(|&x| (x - min) / (max - min))
    //     //     .collect()
        
    //     scores.iter()
    //         .map(|&x| {
    //             if range < f64::EPSILON {
    //                 0.0
    //             } else {
    //                 // Apply softmax-like normalization
    //                 ((x - min) / range).max(0.0).min(1.0)
    //             }
    //         })
    //         .collect()
    // }

    // /// Determines the direction of correlation based on normalized scores
    // /// 
    // /// # Returns
    // /// * 1 for positive correlation (above threshold)
    // /// * -1 for negative correlation (below inverse threshold)
    // /// * 0 for neutral/uncertain correlation
    // fn determine_direction(&self) -> i8 {
    //     // Configurable threshold for determining correlation direction
    //     let threshold = 0.6;
        
    //     // Calculate average normalized score
    //     let avg_score = (self.norm_chi_square + 
    //                     self.norm_mutual_info + 
    //                     self.norm_logistic_coef) / 3.0;
        
    //     // Determine direction based on threshold
    //     if avg_score > threshold {
    //         1  // Strong positive correlation
    //     } else if avg_score < (1.0 - threshold) {
    //         -1  // Strong negative correlation
    //     } else {
    //         0  // Neutral or uncertain correlation
    //     }
    // }
    
    /// Calculates weighted correlation score with configurable weights
    fn calculate_weighted_score(&self) -> f64 {
        // Define weights for each metric
        const CHI_SQUARE_WEIGHT: f64 = 0.4;
        const MUTUAL_INFO_WEIGHT: f64 = 0.4;
        const LOGISTIC_WEIGHT: f64 = 0.2;
        
        (self.norm_chi_square * CHI_SQUARE_WEIGHT +
         self.norm_mutual_info * MUTUAL_INFO_WEIGHT +
         self.norm_logistic_coef * LOGISTIC_WEIGHT)
    }
    
    /// Determines if the feature's correlation is statistically significant
    pub fn is_significant(&self, significance_threshold: f64) -> bool {
        // Check if any metric shows significant correlation
        self.raw_chi_square > significance_threshold ||
        self.raw_mutual_info > significance_threshold ||
        self.raw_logistic_coef > significance_threshold
    }

    /// Determines correlation direction with significance check
    fn determine_direction(&self) -> i8 {
        const SIGNIFICANCE_THRESHOLD: f64 = 0.05;  // Standard statistical significance
        const CORRELATION_THRESHOLD: f64 = 0.6;    // Strength threshold
        
        if !self.is_significant(SIGNIFICANCE_THRESHOLD) {
            return 0;  // Not significant
        }
        
        let avg_score = self.calculate_weighted_score();
        
        if avg_score > CORRELATION_THRESHOLD {
            1
        } else if avg_score < (1.0 - CORRELATION_THRESHOLD) {
            -1
        } else {
            0
        }
    }
}

/// Performs normalized feature correlation analysis on bag-of-words matrix
/// 
/// # Arguments
/// * `bow_matrix` - Document-term matrix
/// * `labels` - Vector of class labels
/// * `stem_dictionary` - Optional dictionary mapping feature indices to tokens
/// 
/// # Returns
/// * Vector of NormalizedFeatureAnalysis results sorted by correlation strength
pub fn analyze_feature_correlations_normalized(
    bow_matrix: &Array2<f64>,
    labels: &[usize],
    stem_dictionary: Option<&[String]>,
) -> Result<Vec<NormalizedFeatureAnalysis>, FeatureSelectionError> {
    // Validate input dimensions
    if bow_matrix.is_empty() || labels.is_empty() {
        return Err(FeatureSelectionError::EmptyInput);
    }
    if bow_matrix.nrows() != labels.len() {
        return Err(FeatureSelectionError::DimensionMismatch);
    }

    let num_features = bow_matrix.ncols();

    // Calculate raw correlation scores
    let chi_square_scores = calculate_chi_square_scores(bow_matrix, labels)?;
    let mutual_info_scores = calculate_mutual_information_scores(bow_matrix, labels)?;
    let logistic_scores = calculate_logistic_regression_scores(bow_matrix, labels)?;

    // Normalize all scores to [0, 1] range
    let chi_square_norm = NormalizedFeatureAnalysis::normalize_scores(&chi_square_scores);
    let mutual_info_norm = NormalizedFeatureAnalysis::normalize_scores(&mutual_info_scores);
    let logistic_norm = NormalizedFeatureAnalysis::normalize_scores(&logistic_scores);

    // Create normalized analysis results
    let mut normalized_results = Vec::with_capacity(num_features);
    for i in 0..num_features {
        let mut analysis = NormalizedFeatureAnalysis {
            // Get token name from dictionary or generate default
            token: stem_dictionary.map_or_else(
                || format!("feature_{}", i),
                |dict| dict[i].clone()
            ),
            feature_index: i,
            
            // Store raw scores
            raw_chi_square: chi_square_scores[i],
            raw_mutual_info: mutual_info_scores[i],
            raw_logistic_coef: logistic_scores[i],
            
            // Store normalized scores
            norm_chi_square: chi_square_norm[i],
            norm_mutual_info: mutual_info_norm[i],
            norm_logistic_coef: logistic_norm[i],
            
            // // Calculate combined score
            // weighted_correlation: (chi_square_norm[i] + 
            //                      mutual_info_norm[i] + 
            //                      logistic_norm[i]) / 3.0,
            // correlation_direction: 0,  // Will be set below
            
            weighted_correlation: 0.0,  // will be set below
            correlation_direction: 0,   // will be set below
        };
        
        // Calculate weighted score
        analysis.weighted_correlation = analysis.calculate_weighted_score();
        analysis.correlation_direction = analysis.determine_direction();
        
        // Calculate direction based on normalized scores
        analysis.correlation_direction = analysis.determine_direction();
        
        normalized_results.push(analysis);
    }

    // Sort by weighted correlation strength
    normalized_results.sort_by(|a, b| b.weighted_correlation
        .partial_cmp(&a.weighted_correlation)
        .unwrap_or(std::cmp::Ordering::Equal));

    Ok(normalized_results)
}

/// Classifier using normalized feature correlations
pub struct ByteClassifier {
    /// Analyzed features with correlation scores
    features: Vec<NormalizedFeatureAnalysis>,
    /// Classification threshold
    threshold: f64,
}

impl ByteClassifier {
    /// Creates a new ByteClassifier with specified threshold
    /// 
    /// # Arguments
    /// * `threshold` - Classification threshold (typically 1.0)
    pub fn new(threshold: f64) -> Self {
        Self {
            features: Vec::new(),
            threshold,
        }
    }

    /// Classifies a document based on correlation scores
    /// 
    /// # Arguments
    /// * `document` - Text to classify
    /// 
    /// # Returns
    /// * `true` if document meets classification threshold
    /// * `false` otherwise
    pub fn classify(&self, document: &str) -> bool {
        let score = self.calculate_document_score(document);
        score >= self.threshold
    }

    /// Calculates correlation score for a document
    /// 
    /// # Arguments
    /// * `document` - Text to analyze
    /// 
    /// # Returns
    /// * Combined correlation score for the document
    fn calculate_document_score(&self, document: &str) -> f64 {
        let mut score = 0.0;
        
        // Sum weighted correlations for each matching feature
        for feature in &self.features {
            if document.contains(&feature.token) {
                // Add or subtract based on correlation direction
                score += feature.weighted_correlation * 
                        (feature.correlation_direction as f64);
            }
        }
        
        score
    }
}

// /// Helper function to print normalized analysis results
// pub fn print_normalized_analysis(
//     results: &[NormalizedFeatureAnalysis],
//     top_n: Option<usize>
// ) {
//     let n = top_n.unwrap_or(results.len());
//     println!("\nTop {} Features by Normalized Correlation:", n);
//     println!("{:-<80}", "");

//     for (i, result) in results.iter().take(n).enumerate() {
//         println!("{}. Token: {}", i + 1, result.token);
//         println!("   Raw Scores:");
//         println!("      Chi-Square: {:.4}", result.raw_chi_square);
//         println!("      Mutual Info: {:.4}", result.raw_mutual_info);
//         println!("      Logistic Coef: {:.4}", result.raw_logistic_coef);
//         println!("   Normalized Scores:");
//         println!("      Chi-Square: {:.4}", result.norm_chi_square);
//         println!("      Mutual Info: {:.4}", result.norm_mutual_info);
//         println!("      Logistic Coef: {:.4}", result.norm_logistic_coef);
//         println!("   Combined Score: {:.4}", result.weighted_correlation);
//         println!("   Correlation Direction: {}", result.correlation_direction);
//         println!("{:-<80}", "");
//     }
// }

pub fn print_normalized_analysis(
    results: &[NormalizedFeatureAnalysis],
    top_n: Option<usize>
) {
    let n = top_n.unwrap_or(results.len());
    println!("\nTop {} Features by Normalized Correlation:", n);
    println!("{:-<80}", "");

    for (i, result) in results.iter().take(n).enumerate() {
        println!("{}. Token: {} (Index: {})", i + 1, result.token, result.feature_index);
        println!("   Raw Scores:");
        println!("      Chi-Square: {:.4} (p-value: {:.4})", 
                result.raw_chi_square, 
                calculate_p_value(result.raw_chi_square, 1));
        println!("      Mutual Info: {:.4}", result.raw_mutual_info);
        println!("      Logistic Coef: {:.4}", result.raw_logistic_coef);
        println!("   Normalized Scores:");
        println!("      Chi-Square: {:.4}", result.norm_chi_square);
        println!("      Mutual Info: {:.4}", result.norm_mutual_info);
        println!("      Logistic Coef: {:.4}", result.norm_logistic_coef);
        println!("   Weighted Score: {:.4}", result.weighted_correlation);
        println!("   Correlation Direction: {} ({})", 
                result.correlation_direction,
                match result.correlation_direction {
                    1 => "Positive",
                    -1 => "Negative",
                    0 => "Neutral/Insignificant",
                    _ => "Unknown"
                });
        println!("   Significant: {}", result.is_significant(0.05));
        println!("{:-<80}", "");
    }
}

/*
fn main() -> io::Result<()> {
    // Example data
    let bow_matrix = arr2(&[
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let labels = vec![0, 1, 0, 1];
    let stem_dictionary = vec![
        "first".to_string(),
        "second".to_string(),
        "third".to_string(),
    ];

    // Perform normalized analysis
    let analysis_results = analyze_feature_correlations_normalized(
        &bow_matrix,
        &labels,
        Some(&stem_dictionary)
    )?;

    // Print results
    print_normalized_analysis(&analysis_results, Some(3));

    // Create classifier
    let classifier = ByteClassifier::new(1.0);
    
    Ok(())
}
*/



// Test with bash: Cargo Test
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::write;
    use tempfile::NamedTempFile;
    use ndarray::arr2;

    #[test]
    fn test_ends_with_edge_cases() {
        let mut stemmer = PorterStemmer::new();
        
        // Test empty string
        stemmer.buffer = vec!['a'];
        stemmer.k = 0;
        stemmer.k0 = 0;
        assert!(stemmer.ends_with(""));
        
        // Test single character
        stemmer.buffer = vec!['a'];
        stemmer.k = 0;
        stemmer.k0 = 0;
        assert!(stemmer.ends_with("a"));
        assert!(!stemmer.ends_with("b"));
        
        // Test short buffer
        stemmer.buffer = vec!['a'];
        stemmer.k = 0;
        stemmer.k0 = 0;
        assert!(!stemmer.ends_with("ab"));  // Should return false, not panic
        
        // Test boundary conditions
        stemmer.buffer = vec!['a', 'b', 'c'];
        stemmer.k = 2;
        stemmer.k0 = 0;
        assert!(stemmer.ends_with("c"));
        assert!(stemmer.ends_with("bc"));
        assert!(stemmer.ends_with("abc"));
        assert!(!stemmer.ends_with("dabc"));  // Should return false, not panic
    }
        
    
    #[test]
    fn test_feature_analysis_printing() {
        let analysis = vec![
            FeatureAnalysis {
                feature_index: 0,
                token: "test_token".to_string(),
                chi_square_value: 0.5,
                mutual_info_score: 0.3,
                logistic_coef: 0.4,
                combined_score: 0.4,
            }
        ];
        
        print_feature_analysis(&analysis, Some(1));
    }

    #[test]
    fn test_save_analysis_csv() -> io::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let output_path = temp_dir.path().join("test_analysis.csv");
        
        let analysis = vec![
            FeatureAnalysis {
                feature_index: 0,
                token: "test_token".to_string(),
                chi_square_value: 0.5,
                mutual_info_score: 0.3,
                logistic_coef: 0.4,
                combined_score: 0.4,
            }
        ];

        save_feature_analysis_csv(&analysis, output_path.to_str().unwrap())?;
        assert!(output_path.exists());
        Ok(())
    }

    #[test]
    fn test_generate_summary() -> io::Result<()> {
        let analysis = vec![
            FeatureAnalysis {
                feature_index: 0,
                token: "test_token".to_string(),
                chi_square_value: 0.5,
                mutual_info_score: 0.3,
                logistic_coef: 0.4,
                combined_score: 0.4,
            }
        ];

        let summary = generate_analysis_summary(&analysis, None)?;
        assert!(!summary.is_empty());
        assert!(summary.contains("Feature Analysis Summary"));
        Ok(())
    }
    
    // comprehensive feature analysis
    #[test]
    fn test_basic_feature_analysis() {
        let bow_matrix = arr2(&[
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let labels = vec![0, 1, 0, 1];
        let stem_dictionary = vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ];

        let result = analyze_feature_correlations(&bow_matrix, &labels, Some(&stem_dictionary));
        assert!(result.is_ok());

        if let Ok(analysis) = result {
            assert_eq!(analysis.len(), 3);
            assert!(analysis[0].combined_score >= analysis[1].combined_score);
            assert!(analysis[1].combined_score >= analysis[2].combined_score);
        }
    }

    #[test]
    fn cfa_test_empty_input() {
        let bow_matrix = Array2::<f64>::zeros((0, 0));
        let labels = vec![];
        
        let result = analyze_feature_correlations(&bow_matrix, &labels, None);
        assert!(matches!(result, Err(FeatureSelectionError::EmptyInput)));
    }

    #[test]
    fn cfa_test_dimension_mismatch() {
        let bow_matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let labels = vec![0, 1, 2];  // More labels than rows
        
        let result = analyze_feature_correlations(&bow_matrix, &labels, None);
        assert!(matches!(result, Err(FeatureSelectionError::DimensionMismatch)));
    }
   
    // GLM
    #[test]
    fn test_logistic_regression_basic() {
        let x = arr2(&[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]);
        let y = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iterations(1000);
            
        let result = model.fit(x.view(), y.view());
        assert!(result.is_ok(), "Failed to fit model: {:?}", result);
    }

    #[test]
    fn test_feature_importance() {
        let bow_matrix = arr2(&[
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let labels = vec![0, 1, 0, 1];

        let result = calculate_logistic_regression_importance(&bow_matrix, &labels);
        assert!(result.is_ok(), "Feature importance calculation failed: {:?}", result);

        if let Ok(importance) = result {
            assert_eq!(importance.len(), 3, "Expected 3 features");
            // Check if importance scores are ordered
            for i in 1..importance.len() {
                assert!(
                    importance[i-1].abs_importance >= importance[i].abs_importance,
                    "Importance scores should be in descending order"
                );
            }
        }
    }

    // Add more test cases
    #[test]
    fn glm_test_empty_input() {
        let x = arr2(&[[0.0; 0]; 0]);
        let y = Array1::<f64>::zeros(0);
        
        let mut model = LogisticRegression::new();
        let result = model.fit(x.view(), y.view());
        assert!(matches!(result, Err(LogisticRegressionError::InvalidInput(_))));
    }

    #[test]
    fn glm_test_dimension_mismatch() {
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let y = Array1::from_vec(vec![1.0]);  // Wrong length
        
        let mut model = LogisticRegression::new();
        let result = model.fit(x.view(), y.view());
        assert!(matches!(result, Err(LogisticRegressionError::DimensionMismatch(_))));
    }
    
    
    // mutual_information
    #[test]
    fn test_mutual_information_basic() {
        let bow_matrix = arr2(&[
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        let labels = vec![0, 0, 1, 1];

        let result = calculate_mutual_information(&bow_matrix, &labels);
        assert!(result.is_ok());

        let scores = result.unwrap();
        assert!(!scores.is_empty());
        assert!(scores[0].mi_score >= 0.0);
    }

    #[test]
    fn test_mutual_information_empty_input() {
        let bow_matrix = Array2::<f64>::zeros((0, 0));
        let labels = vec![];

        let result = calculate_mutual_information(&bow_matrix, &labels);
        assert!(matches!(result, Err(FeatureSelectionError::EmptyInput)));
    }

    #[test]
    fn test_mutual_information_dimension_mismatch() {
        let bow_matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let labels = vec![0, 1, 2];  // More labels than rows

        let result = calculate_mutual_information(&bow_matrix, &labels);
        assert!(matches!(result, Err(FeatureSelectionError::DimensionMismatch)));
    }

    #[test]
    fn test_perfect_correlation() {
        let bow_matrix = arr2(&[
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]);
        let labels = vec![0, 0, 1, 1];

        let result = calculate_mutual_information(&bow_matrix, &labels).unwrap();
        
        // Perfect correlation should have higher MI score
        assert!(result[0].mi_score > 0.5);
    }    
    
    #[test]
    fn test_chi_square_basic() {
        // Create a more distinctive pattern in test data
        let bow_matrix = arr2(&[
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]);
        // Labels that clearly correlate with the features
        let labels = vec![0, 0, 1, 1, 0, 1];
        
        let result = chi_square_feature_selection(&bow_matrix, &labels, 0.05);
        assert!(result.is_ok(), "Chi-square calculation failed");
        
        let correlations = result.unwrap();
        assert!(!correlations.is_empty(), "No significant correlations found");
        
        // Additional assertions to verify the results
        if let Some(first_correlation) = correlations.first() {
            assert!(
                first_correlation.chi_square_value > 0.0,
                "Chi-square value should be positive"
            );
            assert!(
                first_correlation.p_value <= 0.05,
                "P-value should be below significance threshold"
            );
        }
    }

    // Add a new test for threshold behavior
    #[test]
    fn test_chi_square_threshold() {
        let bow_matrix = arr2(&[
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let labels = vec![0, 1, 0, 1];
        
        // Test with different thresholds
        let strict_result = chi_square_feature_selection(&bow_matrix, &labels, 0.01);
        let lenient_result = chi_square_feature_selection(&bow_matrix, &labels, 0.5);
        
        assert!(strict_result.is_ok());
        assert!(lenient_result.is_ok());
        
        let strict_correlations = strict_result.unwrap();
        let lenient_correlations = lenient_result.unwrap();
        
        // Lenient threshold should include more or equal features than strict
        assert!(lenient_correlations.len() >= strict_correlations.len());
    }

    // Add a test for the p-value calculation
    #[test]
    fn test_p_value_calculation() {
        let chi_square = 10.0;
        let df = 1;
        let p_value = calculate_p_value(chi_square, df);
        
        assert!(p_value > 0.0 && p_value < 1.0, "P-value should be between 0 and 1");
        
        // Test that larger chi-square values result in smaller p-values
        let larger_chi_square = 20.0;
        let larger_p_value = calculate_p_value(larger_chi_square, df);
        assert!(larger_p_value < p_value, "Larger chi-square should give smaller p-value");
    }
    #[test]
    fn test_empty_input() {
        let bow_matrix = Array2::<f64>::zeros((0, 0));
        let labels = vec![];
        
        let result = chi_square_feature_selection(&bow_matrix, &labels, 0.05);
        assert!(matches!(result, Err(FeatureSelectionError::EmptyInput)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let bow_matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let labels = vec![0, 1, 2];  // More labels than rows
        
        let result = chi_square_feature_selection(&bow_matrix, &labels, 0.05);
        assert!(matches!(result, Err(FeatureSelectionError::DimensionMismatch)));
    }
    
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
        // Create temporary files with proper scope
        let temp_dir = tempfile::tempdir()?;
        let input_file = NamedTempFile::new()?;
        let output_path = temp_dir.path().join("output.csv");
        let dict_path = temp_dir.path().join("dict.txt");
        
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
            output_path.to_str().unwrap(),
            dict_path.to_str().unwrap(),
            1,
            10,
        )?;

        // Verify dictionary
        let stems = PorterStemmer::read_stem_dictionary(dict_path.to_str().unwrap())?;
        println!("Stems in dictionary: {:?}", stems);
        assert!(stems.contains(&"run".to_string()));
        assert!(stems.contains(&"jump".to_string()));
        
        // Verify BOW matrix
        let reader = BufReader::new(File::open(&output_path)?);
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
        println!("Output lines: {:?}", lines);
        assert!(!lines.is_empty());
        assert!(!lines[0].ends_with(','));
        
        Ok(())
    }
    
    
    
    
    
}  // End of Tests


fn inspect_csv(path: &str) -> io::Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    // Read first few lines
    for (i, line) in reader.lines().take(5).enumerate() {
        println!("Line {}: {:?}", i, line?);
    }
    Ok(())
}

fn main() -> io::Result<()> {
    
    // // // config
    // // // Stopwords option
    // // let config = TokenizerConfig {
    // //     remove_stopwords: true,
    // //     text_column: 0,
    // // };
    
    // // Ensure directories exist
    // ensure_directories()?;
    
    // let mut stemmer = PorterStemmer::new();


    // ///////////////
    // // Single Path
    // ///////////////
    // // let single_file_input_path = "file_targets/corpus.csv";
    // let single_file_input_path = "file_targets/mini.csv";  // thread 'main' panicked at src/main.rs:2606:18:, attempt to subtract with overflow
    // let text_field_index_for_inputpath:usize = 2;
    
    // // Inspect CSV before processing
    // println!("Inspecting .csv structure:");
    // inspect_csv(single_file_input_path)?;    
    
    
    // //////////////////
    // // 1. single word
    // //////////////////
    // print!("1. single word\n");
    // let stemmed = stemmer.stem("running");
    // println!("stemmed = stemmer.stem('running'): {}\n", stemmed); // Outputs: "run"
    
    // // //////////////////////
    // // // 2. single doc string
    // // ////////////////////////
    // // // Process text directly
    // // print!("2. single doc string\n");
    // // let text = "Running and jumping through the fields!";
    // // let stemmed_text = stemmer.process_text(text);
    // // println!("Stemmed text results: {:?}\n", stemmed_text);

    // // //////////////////////
    // // // 3. single doc file
    // // //////////////////////
    // // print!("3. single doc file\n");
    // // let stemmer = PorterStemmer::new();

    // // // Process a document file
    // // let stemmed_words = stemmer.process_file_document("file_targets/test.txt")?;
    // // println!("-> Stemmed words: {:?}", stemmed_words);
    
    // // // Process with frequency counting
    // // let word_frequencies = stemmer.process_documentfile_with_frequencies("file_targets/test.txt")?;
    // // println!("-> Word frequencies: {:?}\n", word_frequencies);
    
    // // print!("3.2 noload single doc file\n");
    // // // Process document file with streaming
    // // let stems = stemmer.noload_process_filedocument_streaming(
    // //     "file_targets/test.txt", // input path
    // //     "output/test_stems.txt",        // output path
    // //     1000, // chunk size
    // // )?;
    // // println!("-> Unique stems written to test_stems.txt");
    // // println!("{:?}", stems);
    
    // // // Process with frequency counting
    // // let frequencies = stemmer.noload_process_documentfile_frequencies_streaming(
    // //     "file_targets/test.txt", // input path
    // //     "output/test_stems.txt",        // output path
    // //     "output/test_frequencies.txt",  // output path
    // //     1000, // chunk size
    // // )?;
    // // println!("-> Stem frequencies written to test_frequencies.txt\n");
    // // println!("{:?}", frequencies);
    
    // // ///////////////////
    // // // 4.1 csv corpus
    // // ///////////////////
    // // print!("4.1 csv corpus\n");
    // // // Process CSV and create document-term stem matrix
    // // stemmer.process_csv_to_bow_matrix(
    // //     "file_targets/test_csv.csv", // read this input
    // //     "output/output_with_bow.csv",       // path to output for results
    // //     "output/stem_dictionary.txt",       // stem-dict output
    // //     0,                           // assuming text is in column 1
    // // )?;
    
    
    // // //////////////////////////////
    // // // 4.2 extra-noload csv corpus
    // // //////////////////////////////
    // // print!("4.2 extra-noload csv corpus\n");
    // // // Process with explicit streaming and chunk size
    // // // Process with explicit streaming and chunk size
    // // stemmer.noload_process_csvtobow_matrix_streaming(
    // //     "file_targets/corpus.csv",
    // //     "output/noload_output_with_bow.csv",
    // //     "output/noload_stem_dictionary.txt",
    // //     0,  // corpus text column index
    // //     1000,
    // // )?;
    
    
    // ////////////////////////////////
    // // 5. Two Phase .csv ~tokenizer
    // ////////////////////////////////
    // // let single_file_input_path = "file_targets/corpus.csv";
    // // let single_file_input_path = "file_targets/train.csv";  // thread 'main' panicked at src/main.rs:2606:18:, attempt to subtract with overflow
    // let tokenizer_dict_path = "output/tokenizer_dict.json";
    // let bow_matrix_path = "output/bow_matrix.csv";

    // // First sweep: Collect stems
    // let mut tokenizer_dict = TokenizerDict::new();
    // tokenizer_dict.first_sweep(
    //     single_file_input_path, // single_file_input_path
    //     text_field_index_for_inputpath, // corpus text column index
    //     )?;
    
    // // Save tokenizer dictionary
    // tokenizer_dict.save_to_json(tokenizer_dict_path)?;

    // // Optional: Load dictionary (demonstrating persistence)
    // let loaded_dict = TokenizerDict::load_from_json(tokenizer_dict_path)?;

    // // Second sweep: Create BOW matrix
    // loaded_dict.second_sweep(
    //     single_file_input_path, // input path
    //     bow_matrix_path, // output path
    //     text_field_index_for_inputpath, // corpus text column index
    // )?;

    // ////////////////////////////////////////////////
    // // 5.2 Stopworks with Two Phase .csv ~tokenizer
    // //////////////////////?????????????????/////////
    // // let single_file_input_path = "file_targets/corpus.csv";
    // // let single_file_input_path = "file_targets/train.csv";  // thread 'main' panicked at src/main.rs:2606:18:, attempt to subtract with overflow
    // let tokenizer_dict_path = "output/tokenizer_dict.json";
    // let bow_matrix_path = "output/bow_matrix.csv";

    // // First sweep with stopword filtering
    // let mut tokenizer_dict = TokenizerDict::new();
    // tokenizer_dict.first_sweep_with_stopwords(
    //     single_file_input_path,
    //     text_field_index_for_inputpath, // corpus text column index
    // )?;
    
    // // Save tokenizer dictionary
    // tokenizer_dict.save_to_json(tokenizer_dict_path)?;

    // // Optional: Load dictionary (demonstrating persistence)
    // let loaded_dict = TokenizerDict::load_from_json(tokenizer_dict_path)?;

    // // Second sweep: Create BOW matrix
    // loaded_dict.second_sweep(
    //     single_file_input_path, // input path
    //     bow_matrix_path, // output path
    //     text_field_index_for_inputpath, // corpus text column index
    // )?;    
    
    
    // //////////////////////////
    // // 6. Multiple .csv files
    // //////////////////////////
    // // Setup input files and output directory
    // let input_files_list = vec![
    //     "file_targets/corpus.csv".to_string(),
    //     "file_targets/corpus2.csv".to_string(),
    //     "file_targets/corpus3.csv".to_string(),
        
    // ];
    
    // let output_dir = "output/multiple_files";
    
    // // Create and run the multi-file processor
    // let processor = MultiFileProcessor::new(
    //     input_files_list,
    //     output_dir.to_string(),
    //     0,  // text column index
    // );
    
    // processor.process_all_files()?;

    // //////////////////////////
    // // 7. TF-IDF 
    // //////////////////////////
    // // Process with TF-IDF scores
    // let tokenizer_dict = TokenizerDict::load_from_json(tokenizer_dict_path)?;

    // tokenizer_dict.second_sweep_with_tfidf(
    //     single_file_input_path,
    //     "output/bow_matrix_with_tfidf.csv",
    //     text_field_index_for_inputpath, // text column index
    // )?;
    
    
    // /////////////////////////
    // // optional / inspection
    // /////////////////////////
    // // read the stem dictionary:
    // // Read and display the stems from the output directory
    // let stems = PorterStemmer::read_stem_dictionary(
    //     "output/noload_stem_dictionary.txt", // output path
    // )?;
    // println!("Stems used in the BOW matrix:");
    // for stem in stems {
    //     println!("{}", stem);
    // }
    
    
    // ////////////////
    // // Stat Quest!!
    // ////////////////
    // // Example data
    // let bow_matrix = arr2(&[
    //     [1.0, 0.0, 1.0],
    //     [0.0, 1.0, 0.0],
    //     [1.0, 1.0, 0.0],
    //     [0.0, 0.0, 1.0],
    // ]);
    // let labels = vec![0, 1, 0, 1];
    // let stem_dictionary = vec![
    //     "first".to_string(),
    //     "second".to_string(),
    //     "third".to_string(),
    // ];

    // // Perform feature selection
    // let correlations = chi_square_feature_selection(&bow_matrix, &labels, 0.05)?;

    // // Print results
    // print_feature_correlations(&correlations, Some(3), Some(&stem_dictionary));

    // // Original stemmer code...
    // let mut stemmer = PorterStemmer::new();
    // let stemmed = stemmer.stem("running");
    // println!("stemmed = stemmer.stem('running'): {}\n", stemmed);


    // //////////////////////
    // // mutual_information
    // //////////////////////
    // let bow_matrix = arr2(&[
    //     [1.0, 0.0, 1.0],
    //     [0.0, 1.0, 0.0],
    //     [1.0, 1.0, 0.0],
    //     [0.0, 0.0, 1.0],
    // ]);
    // let labels = vec![0, 1, 0, 1];
    // let stem_dictionary = vec![
    //     "first".to_string(),
    //     "second".to_string(),
    //     "third".to_string(),
    // ];

    // // Calculate mutual information
    // let mi_scores = calculate_mutual_information(&bow_matrix, &labels)
    //     .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // // Print results
    // print_mi_feature_importance(&mi_scores, Some(&stem_dictionary), Some(3));


    // //////////////////////
    // // GLM
    // //////////////////////
    
    
    // ////////////////////////////////
    // // analyze_feature_correlations
    // ////////////////////////////////
    // let bow_matrix = arr2(&[
    //     [1.0, 0.0, 1.0],
    //     [0.0, 1.0, 0.0],
    //     [1.0, 1.0, 0.0],
    //     [0.0, 0.0, 1.0],
    // ]);
    // let labels = vec![0, 1, 0, 1];
    // let stem_dictionary = vec![
    //     "first".to_string(),
    //     "second".to_string(),
    //     "third".to_string(),
    // ];

    // let analysis_results = analyze_feature_correlations(
    //     &bow_matrix,
    //     &labels,
    //     Some(&stem_dictionary)
    // )?;

    // print_feature_analysis(&analysis_results, Some(3));
    
    
    // // Your analysis results
    // let analysis_results = vec![
    //     FeatureAnalysis {
    //         feature_index: 0,
    //         token: "example_token".to_string(),
    //         chi_square_value: 0.75,
    //         mutual_info_score: 0.45,
    //         logistic_coef: 0.60,
    //         combined_score: 0.60,
    //     },
    //     // ... more results
    // ];

    // // Generate reports
    // analyze_and_report(analysis_results, "output/analysis")?;
    
    
    ///////////////
    // Normalized
    ///////////////
    // Example data
    let bow_matrix = arr2(&[
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let labels = vec![0, 1, 0, 1];
    let stem_dictionary = vec![
        "first".to_string(),
        "second".to_string(),
        "third".to_string(),
    ];

    // Perform normalized analysis
    let analysis_results = analyze_feature_correlations_normalized(
        &bow_matrix,
        &labels,
        Some(&stem_dictionary)
    )?;

    // Print results
    print_normalized_analysis(&analysis_results, Some(3));

    // Create classifier
    let classifier = ByteClassifier::new(1.0);
    
    
    
    // Finis
    Ok(())
    
}

/*
normal_byte_classer_rust
byteclasser
ngram byte target modeling
byteclasser?
labels_to_byte_matrix_classifier
doc_term_porterstem_matrix_rust
labeled_dataset_to_by_scan_classifier_rust
byte_tokenizer_rust

Potential future items:
- Add TF-IDF calculation (in progress)
- Support for more advanced tokenization
- Configurable stop word removal
- Parallel processing of large datasets
- optional lematizer perhaps from NLTK or other open-source

maybe add other input formats:
dir of docs
jsonl
json


what format is best (or good enough) for efficient storing the list of stem-tokens
to be reloaded to populate the csv file?

next step / TODO:
- remove unwrap or any code not safe for production use
- make all error messages clear: identify the function producing the error
- check that stopwords can be enabled for multi-document processing


normalized byte classer scoring

1. get correlation analysis with normalized results
an overall normalized correclation score becomes the weight of that target (byte ideally)
(reach goal: with possible negative scored values for negative correlations, e.g. terms that only correlate with not-hotdog,
    as meat terms for vegan food perhaps)

2. Run byteclasser with threshold... 
default threshold of 1, 

(experiment with recipe etc tests)


a normalized overal standard correclation score can be added on to (at least some) of the existing correlation tests?

again, the target is just like weighted matching,
weaker correlated targets will get a low weight (normalized correlation score), strongly correclated targets will get a higher score. and hopefully with real example testing some standard range of threshold (maybe simple: 1) will work to flag that class for that document.   the full normal analysis needs to be there for transparancy, but a normalized score should also be able to be added on without much code change. 

my first thought it doubleing the FeatureAnalysis fields: 



/// Extend the existing FeatureAnalysis with normalized metrics
#[derive(Debug, Clone)]
pub struct FeatureAnalysis {

    pub standard_token: String,
// string byte?

    // Existing fields
    pub standard_feature_index: usize,
    pub standard_chi_square_value: f64,
    pub standard_mutual_info_score: f64,
    pub standard_logistic_coef: f64,
// combining different statistical tests without normalization would be meaningless and mathematically incorrect

//normalized_
    pub normalized_feature_index: usize,
    pub normalized_chi_square_value: f64,
    pub normalized_mutual_info_score: f64,
    pub normalized_ logistic_coef: f64,

// combined
    pub normalized_combined_score: f64,

no colliding names

Also, there should be a plan for how to use tfid numbers...

or to have an option of raw stem frequency vs. ftidf perhaps....


*/ 

