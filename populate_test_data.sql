USE compound_db;

# Library
insert into library (name) values ('DSSTox');
insert into library (name) values ('ToxCast');
insert into library (name, doi) values ('HMDB', '10.1093/nar/gkx1089');
insert into library (name) values ('Dorrenstein');

# Category
insert into category (name)
values 
('Name'),
('SMILES'),
('InChI'),
('Formula'),
('[M+H] CCS'),
('[M+Na] CCS'),
('[M-H] CCS');