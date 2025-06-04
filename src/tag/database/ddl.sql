CREATE TABLE regulations (
    id VARCHAR(20) PRIMARY KEY,
    url TEXT,
    download_link TEXT,
    download_name TEXT,
    title TEXT,
    about TEXT,
    type TEXT,
    short_type TEXT,
    amendment TEXT,
    number VARCHAR(10),
    year INT,
    institution TEXT,
    issue_place TEXT,
    issue_date DATE,
    effective_date DATE
);

CREATE TABLE subjects (
    id VARCHAR(20),
    subject TEXT,
    CONSTRAINT fk_subjects_regulations FOREIGN KEY (id) REFERENCES regulations(id) ON DELETE CASCADE
);

CREATE TABLE status (
    id VARCHAR(20),
    repealed TEXT,
    repeal TEXT,
    amended TEXT,
    amend TEXT,
    CONSTRAINT fk_status_regulations FOREIGN KEY (id) REFERENCES regulations(id) ON DELETE CASCADE
);

CREATE TABLE articles (
    id VARCHAR(20) PRIMARY KEY,
    regulation_id VARCHAR(20),
    chapter_number VARCHAR(10),
    chapter_about TEXT,
    article_number VARCHAR(10),
    text TEXT,
    status VARCHAR(20) DEFAULT 'Effective', -- 'Effective' atau 'Ineffective'
    CONSTRAINT fk_articles_regulations FOREIGN KEY (regulation_id) REFERENCES regulations(id) ON DELETE CASCADE
);

CREATE TABLE definitions (
    id VARCHAR(20) PRIMARY KEY,
    regulation_id VARCHAR(20),
    name TEXT,
    definition TEXT,
    CONSTRAINT fk_definitions_regulations FOREIGN KEY (regulation_id) REFERENCES regulations(id) ON DELETE CASCADE
);

CREATE TABLE regulation_relations (
    from_regulation_id VARCHAR(20),
    to_regulation_id VARCHAR(20),
    relation_type TEXT, -- contoh: "mengubah", "mencabut", "merujuk"
    CONSTRAINT pk_regulation_relations PRIMARY KEY (from_regulation_id, to_regulation_id, relation_type),
    CONSTRAINT fk_from_regulation FOREIGN KEY (from_regulation_id) REFERENCES regulations(id) ON DELETE CASCADE,
    CONSTRAINT fk_to_regulation FOREIGN KEY (to_regulation_id) REFERENCES regulations(id) ON DELETE CASCADE
);

CREATE TABLE article_relations (
    from_article_id VARCHAR(20),
    to_article_id VARCHAR(20),
    relation_type TEXT, -- contoh: "sebelumnya", "berikutnya", "merujuk", "mengubah"
    CONSTRAINT pk_article_relations PRIMARY KEY (from_article_id, to_article_id, relation_type),
    CONSTRAINT fk_from_article FOREIGN KEY (from_article_id) REFERENCES articles(id) ON DELETE CASCADE,
    CONSTRAINT fk_to_article FOREIGN KEY (to_article_id) REFERENCES articles(id) ON DELETE CASCADE
);
