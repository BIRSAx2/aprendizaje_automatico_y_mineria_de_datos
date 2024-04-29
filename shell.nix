with import <nixpkgs> {};

( let in python311.withPackages (ps: with ps; 
    [ 
    numpy pandas scipy matplotlib sklearn-deap
    ]
  )
).env
