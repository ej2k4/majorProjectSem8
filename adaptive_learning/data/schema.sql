-- ============================================================
--  Adaptive Learning Platform — Database Schema
--  Audience: Ages 4–7 | Modules: Math, Science, Social
-- ============================================================

-- ------------------------------------------------------------
-- 1. STUDENTS
-- ------------------------------------------------------------
CREATE TABLE students (
    student_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             VARCHAR(100),
    age              SMALLINT CHECK (age BETWEEN 3 AND 8),
    created_at       TIMESTAMP DEFAULT NOW(),
    -- Cold-start flag: FALSE until 5+ events recorded
    warmup_complete  BOOLEAN DEFAULT FALSE,
    -- Cluster assigned by K-Means (NULL until enough data)
    cluster_label    SMALLINT,          -- 0=fast, 1=consistent, 2=distracted, 3=mixed
    cluster_updated_at TIMESTAMP
);

-- ------------------------------------------------------------
-- 2. CONTENT LIBRARY
-- ------------------------------------------------------------
CREATE TABLE questions (
    question_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module           VARCHAR(20) NOT NULL CHECK (module IN ('math', 'science', 'social')),
    topic            VARCHAR(50) NOT NULL,
    -- e.g. math: 'addition','subtraction','multiplication','division'
    --      science: 'planets','environments','mechanics'
    --      social: 'etiquette','manners','behaviour'
    difficulty       SMALLINT NOT NULL CHECK (difficulty BETWEEN 1 AND 3),
    -- 1=easy, 2=medium, 3=hard  (narrow range for ages 4–7)
    question_text    TEXT NOT NULL,
    correct_answer   TEXT NOT NULL,
    answer_options   JSONB,             -- for MCQ: ["A","B","C"]
    visual_asset_url TEXT,              -- image/emoji aid (critical for this age group)
    created_at       TIMESTAMP DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 3. GAME SESSIONS
-- ------------------------------------------------------------
CREATE TABLE sessions (
    session_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id       UUID REFERENCES students(student_id),
    module           VARCHAR(20) NOT NULL,
    started_at       TIMESTAMP DEFAULT NOW(),
    ended_at         TIMESTAMP,
    total_questions  SMALLINT DEFAULT 0,
    total_correct    SMALLINT DEFAULT 0,
    session_accuracy NUMERIC(5,4),      -- computed on close: correct/total
    avg_response_sec NUMERIC(6,2),
    engagement_score NUMERIC(5,4)       -- 0–1, derived from timing patterns
);

-- ------------------------------------------------------------
-- 4. QUESTION EVENTS  (core ML training table)
-- ------------------------------------------------------------
CREATE TABLE question_events (
    event_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID REFERENCES sessions(session_id),
    student_id          UUID REFERENCES students(student_id),
    question_id         UUID REFERENCES questions(question_id),

    -- ── Raw inputs ──
    module              VARCHAR(20),
    topic               VARCHAR(50),
    difficulty          SMALLINT,
    response_time_sec   NUMERIC(6,2),   -- time from question shown → answer tapped
    answer_given        TEXT,
    is_correct          BOOLEAN,
    attempt_number      SMALLINT,        -- 1st try, 2nd try on same question

    -- ── Derived features (computed at insert time) ──
    past_accuracy_topic NUMERIC(5,4),   -- student's historic accuracy on this topic
    past_accuracy_module NUMERIC(5,4),  -- student's historic accuracy on this module
    attempts_on_topic   SMALLINT,       -- total prior attempts on this topic
    session_number      SMALLINT,       -- how many sessions this student has had
    time_since_last_sec NUMERIC(10,2),  -- seconds since student's last event

    -- ── ML outputs (written back after prediction) ──
    predicted_correct   NUMERIC(5,4),   -- model's P(correct) before question shown
    recommended_difficulty SMALLINT,    -- what the adaptive engine chose next
    cluster_at_time     SMALLINT,       -- student's cluster when event occurred

    created_at          TIMESTAMP DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 5. REWARDS
-- ------------------------------------------------------------
CREATE TABLE rewards (
    reward_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id   UUID REFERENCES students(student_id),
    session_id   UUID REFERENCES sessions(session_id),
    reward_type  VARCHAR(30),   -- 'xp', 'badge', 'streak'
    reward_value JSONB,         -- e.g. {"xp": 10} or {"badge": "star_collector"}
    awarded_at   TIMESTAMP DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 6. RL POLICY TABLE  (Q-table for Q-Learning)
-- ------------------------------------------------------------
CREATE TABLE q_table (
    state_key    VARCHAR(30) PRIMARY KEY,
    -- format: "{accuracy_bucket}_{difficulty}_{cluster}"
    -- e.g. "high_2_0"  (high accuracy, difficulty 2, cluster 0)
    action_easy  NUMERIC(8,4) DEFAULT 0,
    action_mid   NUMERIC(8,4) DEFAULT 0,
    action_hard  NUMERIC(8,4) DEFAULT 0,
    visit_count  INTEGER DEFAULT 0,
    updated_at   TIMESTAMP DEFAULT NOW()
);

-- ------------------------------------------------------------
-- INDEXES
-- ------------------------------------------------------------
CREATE INDEX idx_events_student    ON question_events(student_id);
CREATE INDEX idx_events_session    ON question_events(session_id);
CREATE INDEX idx_events_module     ON question_events(module, topic);
CREATE INDEX idx_events_created    ON question_events(created_at);
CREATE INDEX idx_sessions_student  ON sessions(student_id);