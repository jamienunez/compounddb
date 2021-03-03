DROP DATABASE IF EXISTS compound_db;
CREATE DATABASE compound_db;

USE compound_db;

CREATE TABLE Library (
  name VARCHAR(255) NOT NULL,
  doi VARCHAR(255),
  last_update DATE,
  PRIMARY KEY (name),
  UNIQUE(name)
);
    
CREATE TABLE Compound (
  cpd_id INT UNSIGNED NOT NULL,
  inchi_key VARCHAR(27) NOT NULL,
  # smiles MEDIUMTEXT NOT NULL,  # MEDIUMTEXT holds up to 16,777,215 char
  PRIMARY KEY (cpd_id),
  UNIQUE(inchi_key)  # Look into setting smiles as the unique key. Can't due to size limit.
);

CREATE TABLE LibraryEntry (
  lib_id VARCHAR(20) NOT NULL,
  lib_name VARCHAR(255) NOT NULL,
  cpd_id INT UNSIGNED NOT NULL,
  UNIQUE(lib_id, lib_name)
);

ALTER TABLE LibraryEntry
	ADD CONSTRAINT fk_LibraryEntry_libName_Library_Name
FOREIGN KEY (lib_name) REFERENCES Library (name)
	ON DELETE CASCADE;

ALTER TABLE LibraryEntry
	ADD CONSTRAINT fk_LibraryEntry_cpdId_Compound_cpdId
FOREIGN KEY (cpd_id) REFERENCES Compound (cpd_id)
	ON DELETE CASCADE;

CREATE TABLE Category (
  #category_id int(64) UNSIGNED NOT NULL AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  PRIMARY KEY(name),
  UNIQUE(name)
);

CREATE TABLE Property (
  property_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  cpd_id INT UNSIGNED NOT NULL,
  category VARCHAR(255) NOT NULL,
  value MEDIUMTEXT NOT NULL, # Look more into this, desired: get data type/limit from category
  PRIMARY KEY (property_id),
  UNIQUE(cpd_id, category)  # Will need to change to allow multiple of the same property, diff methods
);

ALTER TABLE Property
	ADD CONSTRAINT fk_Property_cpdId_Compound_cpdId
FOREIGN KEY (cpd_id) REFERENCES Compound (cpd_id)
	ON DELETE CASCADE;

ALTER TABLE Property
	ADD CONSTRAINT fk_Property_categoryId_Category_categoryId
FOREIGN KEY (category) REFERENCES Category (name)
	ON DELETE CASCADE;

CREATE TABLE PropertyMetadata (
  property_id INT UNSIGNED NOT NULL,
  name VARCHAR(255) NOT NULL,
  value VARCHAR(255) NOT NULL,
  UNIQUE(property_id, name)
);

ALTER TABLE PropertyMetadata
	ADD CONSTRAINT fk_PropertyMetadata_propertyId_Property_propertyId
FOREIGN KEY (property_id) REFERENCES Property (property_id)
	ON DELETE CASCADE;

CREATE TABLE Mass (
  cpd_id INT UNSIGNED NOT NULL,
  value float(16) NOT NULL, # Look more into this, desired: get data type/limit from category
  PRIMARY KEY (cpd_id)
);

ALTER TABLE Mass
	ADD CONSTRAINT fk_Mass_cpdId_Compound_cpdId
FOREIGN KEY (cpd_id) REFERENCES Compound (cpd_id)
	ON DELETE CASCADE;

CREATE TABLE MS2_Spectra (
	cpd_id INT UNSIGNED NOT NULL,
    mode BOOLEAN,
    voltage TINYINT,
    spectra_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
	PRIMARY KEY (spectra_id)
);

ALTER TABLE MS2_Spectra
	ADD CONSTRAINT fk_Spectra_cpdId_Compound_cpdId
FOREIGN KEY (cpd_id) REFERENCES Compound (cpd_id)
	ON DELETE CASCADE;

CREATE TABLE Fragment (
	spectra_id INT UNSIGNED NOT NULL,
    mass float(16),
	relative_intensity float(16),
    UNIQUE(spectra_id, mass)
);

ALTER TABLE Fragment
	ADD CONSTRAINT fk_Fragment_spectra_id_MS2_Spectra_spectra_id
FOREIGN KEY (spectra_id) REFERENCES MS2_Spectra (spectra_id)
	ON DELETE CASCADE;