# Validation Proof Guide
## How to Validate Story Completion

**Purpose**: Ensure every story has concrete, verifiable proof of completion before marking as DONE.

**Principle**: "If it's not documented and validated, it's not done."

---

## Table of Contents
1. [Validation Philosophy](#validation-philosophy)
2. [Validation Types](#validation-types)
3. [Evidence Storage](#evidence-storage)
4. [Phase-Specific Validation](#phase-specific-validation)
5. [Validation Checklist Templates](#validation-checklist-templates)
6. [Acceptance Criteria](#acceptance-criteria)

---

## Validation Philosophy

### Three Pillars of Validation

1. **Functional**: Does it work as intended?
2. **Documented**: Can someone else understand and reproduce it?
3. **Tested**: Is there proof it works correctly?

### When is a Story DONE?

✅ **DONE = ALL of the following**:
- [ ] All acceptance criteria met
- [ ] Definition of Done checklist complete
- [ ] Validation proof documented
- [ ] Evidence saved in `validation/` folder
- [ ] PROGRESS_TRACKER.md updated
- [ ] Git commit made with story ID

❌ **NOT DONE = ANY of the following**:
- Acceptance criteria incomplete
- No validation evidence
- Tests failing
- Documentation missing
- Can't demonstrate it working

---

## Validation Types

### 1. Code Validation

**When**: Any story involving code implementation

**Proof Required**:
- [ ] Code runs without errors
- [ ] Output matches expected results
- [ ] Unit tests pass
- [ ] Code coverage meets threshold (80%+)
- [ ] No linting errors
- [ ] Type checking passes

**Evidence Format**:
```bash
# Run the implementation
python src/models/transformer.py > validation/phase2/story-2.X-transformer-output.txt

# Run tests
pytest tests/test_transformer.py -v > validation/phase2/story-2.X-transformer-tests.txt

# Check coverage
pytest --cov=src tests/ --cov-report=term > validation/phase2/story-2.X-coverage.txt

# Generate HTML coverage report
pytest --cov=src tests/ --cov-report=html
# Screenshot: validation/phase2/story-2.X-coverage-report.png

# Lint check
flake8 src/models/transformer.py > validation/phase2/story-2.X-lint.txt

# Type check
mypy src/models/transformer.py > validation/phase2/story-2.X-typecheck.txt
```

**Example Evidence**:
```
validation/phase2/
├── story-2.1-transformer-output.txt      # Successful run
├── story-2.1-transformer-tests.txt       # All tests pass
├── story-2.1-coverage.txt                # 85% coverage
├── story-2.1-coverage-report.png         # Visual coverage
├── story-2.1-lint.txt                    # No errors
└── story-2.1-typecheck.txt               # Type check passed
```

---

### 2. Documentation Validation

**When**: Any story involving writing documentation

**Proof Required**:
- [ ] Word count meets minimum
- [ ] All sections complete
- [ ] Markdown renders correctly
- [ ] Code examples work (if applicable)
- [ ] Links are valid
- [ ] Images/diagrams display correctly

**Evidence Format**:
```bash
# Word count
wc -w 01-foundations/01_attention_basics.md > validation/phase1/story-1.1-word-count.txt

# Check all sections present
grep "^##" 01-foundations/01_attention_basics.md > validation/phase1/story-1.1-sections.txt

# Test code examples (if any)
python -c "exec(open('01-foundations/attention_example.py').read())" > validation/phase1/story-1.1-code-test.txt

# GitHub render screenshot
# validation/phase1/story-1.1-github-render.png
```

**Example Evidence**:
```
validation/phase1/
├── story-1.1-word-count.txt          # 1543 words (>1500 ✓)
├── story-1.1-sections.txt            # All 5 sections present
├── story-1.1-code-test.txt           # Code examples run successfully
└── story-1.1-github-render.png       # Renders correctly on GitHub
```

---

### 3. Training Validation

**When**: Stories involving model training

**Proof Required**:
- [ ] Model trains without errors
- [ ] Loss decreases over epochs
- [ ] Target accuracy/metric achieved
- [ ] Training logs saved
- [ ] Model checkpoint saved
- [ ] Can load and infer from checkpoint

**Evidence Format**:
```bash
# Train model
python scripts/train_transformer.py --epochs 10 > validation/phase2/story-2.8-training-log.txt

# Save training plot
# Plot is automatically saved to results/transformer/training_curve.png

# Test inference from checkpoint
python scripts/test_inference.py --checkpoint models/transformer_epoch10.pt > validation/phase2/story-2.8-inference-test.txt

# Record metrics
cat results/transformer/metrics.json > validation/phase2/story-2.8-metrics.json
```

**Metrics to Capture**:
- Final loss value
- Final accuracy/F1/metric
- Training time
- GPU memory usage
- Number of parameters

**Example Evidence**:
```
validation/phase2/
├── story-2.8-training-log.txt        # Complete training log
├── story-2.8-training-curve.png      # Loss/accuracy plot
├── story-2.8-inference-test.txt      # Inference works
└── story-2.8-metrics.json            # Accuracy: 82% (>80% ✓)
```

---

### 4. API/Deployment Validation

**When**: Stories involving API deployment

**Proof Required**:
- [ ] Service is accessible
- [ ] HTTPS working (if applicable)
- [ ] Health check endpoint responds
- [ ] API returns correct responses
- [ ] Error handling works
- [ ] Authentication works (if implemented)

**Evidence Format**:
```bash
# Test health endpoint
curl https://your-project.com/health > validation/phase7/story-7.X-health.txt

# Test prediction endpoint
curl -X POST https://your-project.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test"}' > validation/phase7/story-7.X-predict.txt

# Test error handling
curl https://your-project.com/api/v1/predict > validation/phase7/story-7.X-error-handling.txt

# Screenshot of working UI (if applicable)
# validation/phase7/story-7.X-ui-working.png

# Screenshot of API docs (Swagger)
# validation/phase7/story-7.X-api-docs.png
```

**Example Evidence**:
```
validation/phase7/
├── story-7.5-health.txt              # {"status": "healthy"}
├── story-7.5-predict.txt             # Correct prediction returned
├── story-7.5-error-handling.txt      # Proper error message
├── story-7.5-ui-working.png          # UI loads and works
└── story-7.5-api-docs.png            # Swagger docs accessible
```

---

### 5. Performance/Benchmark Validation

**When**: Stories involving performance testing

**Proof Required**:
- [ ] Benchmark scripts run successfully
- [ ] Metrics collected
- [ ] Results documented in tables/charts
- [ ] Comparisons made (if applicable)
- [ ] Meets performance targets

**Evidence Format**:
```bash
# Run benchmark
python 04-comparisons/training_comparison.py > validation/phase4/story-4.X-benchmark-log.txt

# Results saved automatically to
# results/benchmarks/training_comparison.csv
# results/benchmarks/plots/training_time.png

# Copy results
cp results/benchmarks/training_comparison.csv validation/phase4/story-4.X-results.csv
cp results/benchmarks/plots/training_time.png validation/phase4/story-4.X-plot.png
```

**Example Evidence**:
```
validation/phase4/
├── story-4.5-benchmark-log.txt       # Benchmark completed
├── story-4.5-results.csv             # All metrics collected
├── story-4.5-plot.png                # Visualization
└── story-4.5-analysis.md             # Interpretation of results
```

---

### 6. Load Testing Validation

**When**: Stories involving scalability testing

**Proof Required**:
- [ ] Load test completes
- [ ] Target RPS achieved
- [ ] Error rate acceptable (<1%)
- [ ] Latency targets met
- [ ] Resource usage documented

**Evidence Format**:
```bash
# Run Locust test
locust -f 08-scalability/load_testing/locust_test.py --headless \
  --users 1000 --spawn-rate 10 --run-time 5m \
  --html validation/phase8/story-8.X-locust-report.html

# Export CSV results
# Locust automatically generates CSV files

# Screenshot of Locust dashboard
# validation/phase8/story-8.X-locust-dashboard.png

# Resource monitoring during test
# validation/phase8/story-8.X-resource-usage.png
```

**Metrics to Capture**:
- Requests per second achieved
- Average latency (p50, p95, p99)
- Error rate
- CPU/Memory usage
- Throughput

**Example Evidence**:
```
validation/phase8/
├── story-8.3-locust-report.html      # Full test report
├── story-8.3-locust-dashboard.png    # Dashboard screenshot
├── story-8.3-resource-usage.png      # CPU/memory graphs
└── story-8.3-results.csv             # Raw data: 1024 RPS achieved ✓
```

---

### 7. Self-Test/Knowledge Validation

**When**: Stories involving learning/understanding

**Proof Required**:
- [ ] Can explain concept without notes
- [ ] Pass self-quiz (80%+ correct)
- [ ] Can implement from memory (for code)
- [ ] Record explanation (video/audio)

**Evidence Format**:
```bash
# Record yourself explaining (2-3 minutes)
# Save to: validation/phase1/story-1.1-explanation.mp4 or .m4a

# Take self-quiz and save results
# validation/phase1/story-1.1-quiz-results.txt

# Draw architecture from memory
# Scan or photo: validation/phase1/story-1.1-hand-drawn.jpg
```

**Quiz Example** (for Attention Mechanism):
```
1. What problem does attention solve?
2. What are Q, K, V in attention?
3. Why do we scale by sqrt(d_k)?
4. What is multi-head attention?
5. How does attention differ from RNNs?
...
(10 questions total)

Score: 9/10 (90%) ✓
```

**Example Evidence**:
```
validation/phase1/
├── story-1.1-explanation.mp4         # 3-min explanation
├── story-1.1-quiz-results.txt        # 9/10 correct
└── story-1.1-hand-drawn.jpg          # Architecture diagram
```

---

## Evidence Storage

### Directory Structure
```
validation/
├── phase0/
│   ├── story-0.1-directory-tree.txt
│   ├── story-0.1-file-list.txt
│   ├── story-0.1-directory-structure.png
│   ├── story-0.2-python-version.txt
│   ├── story-0.2-installed-packages.txt
│   ├── story-0.2-import-test.txt
│   └── ...
├── phase1/
│   ├── story-1.1-word-count.txt
│   ├── story-1.1-attention-output.txt
│   ├── story-1.1-explanation-video.mp4
│   ├── story-1.1-quiz-results.txt
│   └── ...
├── phase2/
│   ├── story-2.1-transformer-output.txt
│   ├── story-2.1-transformer-tests.txt
│   ├── story-2.1-coverage.txt
│   └── ...
└── ...
```

### Naming Convention
**Format**: `story-{phase}.{story}-{description}.{ext}`

**Examples**:
- `story-0.1-directory-tree.txt`
- `story-2.8-training-log.txt`
- `story-4.5-benchmark-results.csv`
- `story-7.10-deployment-screenshot.png`

---

## Phase-Specific Validation

### Phase 0: Project Setup

**Story 0.1**: Directory Structure
```bash
tree -L 3 > validation/phase0/story-0.1-directory-tree.txt
ls -la > validation/phase0/story-0.1-file-list.txt
# Screenshot of directory in Finder/Explorer
```

**Story 0.2**: Python Environment
```bash
source venv/bin/activate
python --version > validation/phase0/story-0.2-python-version.txt
pip list > validation/phase0/story-0.2-installed-packages.txt
python -c "import torch; import transformers; print('OK')" > validation/phase0/story-0.2-import-test.txt
```

**Story 0.4**: Git & GitHub
```bash
git log --oneline > validation/phase0/story-0.4-git-log.txt
git remote -v > validation/phase0/story-0.4-git-remote.txt
# Screenshot of GitHub repo page
```

**Story 0.5**: CI/CD Pipeline
- Screenshot of GitHub Actions passing
- Copy of workflow file
- Badge showing "passing" in README

---

### Phase 1: Foundations

**For Each Theory Document**:
```bash
# Word count
wc -w 01-foundations/XX_document.md

# Test code examples
python 01-foundations/examples/code_example.py

# Self-test
# Record yourself explaining concept
# Take quiz and save results
```

---

### Phase 2-3: Implementations

**For Each Implementation Story**:
```bash
# Run implementation
python src/models/transformer.py

# Run tests
pytest tests/test_transformer.py -v --cov

# Train small model
python scripts/train.py --quick-test

# Save all outputs
```

---

### Phase 4: Benchmarks

**For Each Benchmark Story**:
```bash
# Run benchmark script
python 04-comparisons/benchmark_script.py

# Verify outputs in results/
ls -l results/benchmarks/

# Check plots generated
ls -l results/benchmarks/plots/

# Save copies to validation/
```

---

### Phase 7-8: Deployment & Scale

**Deployment Stories**:
```bash
# Test endpoints
curl https://your-project.com/health
curl -X POST https://your-project.com/api/v1/predict -d '{...}'

# Screenshot of deployed service
# Screenshot of monitoring dashboard
```

**Load Testing Stories**:
```bash
# Run load test
locust -f load_test.py --headless --users 1000

# Save report
# Screenshot of results
# Document metrics achieved
```

---

## Validation Checklist Templates

### Code Implementation Story
```markdown
## Validation Checklist: Story X.X

### Functional Validation
- [ ] Code runs without errors
- [ ] Output is correct
- [ ] Edge cases handled

### Test Validation
- [ ] Unit tests written
- [ ] All tests pass
- [ ] Coverage ≥80%

### Code Quality
- [ ] No linting errors (flake8)
- [ ] Type checking passes (mypy)
- [ ] Code formatted (black)

### Documentation
- [ ] Docstrings complete
- [ ] README updated (if needed)
- [ ] Examples included

### Evidence Saved
- [ ] Output log
- [ ] Test results
- [ ] Coverage report
- [ ] Screenshots (if applicable)

### PROGRESS_TRACKER Updated
- [ ] Story marked as complete
- [ ] Validation proof noted
- [ ] Metrics updated
```

---

### Documentation Story
```markdown
## Validation Checklist: Story X.X

### Content Validation
- [ ] Word count ≥ target
- [ ] All sections complete
- [ ] Examples work
- [ ] Math/equations correct

### Format Validation
- [ ] Markdown renders correctly
- [ ] Links work
- [ ] Images display
- [ ] Code blocks formatted

### Quality Validation
- [ ] Technical accuracy verified
- [ ] Clear and understandable
- [ ] No spelling/grammar errors

### Knowledge Validation
- [ ] Can explain without notes
- [ ] Self-quiz passed (≥80%)
- [ ] Explanation recorded

### Evidence Saved
- [ ] Word count output
- [ ] GitHub render screenshot
- [ ] Quiz results
- [ ] Explanation video/audio
```

---

### Training Story
```markdown
## Validation Checklist: Story X.X

### Training Validation
- [ ] Training completes without errors
- [ ] Loss decreases
- [ ] Target metric achieved
- [ ] Training time documented

### Model Validation
- [ ] Checkpoint saved
- [ ] Can load checkpoint
- [ ] Inference works
- [ ] Model size documented

### Performance Validation
- [ ] Accuracy/metric meets target
- [ ] GPU memory usage acceptable
- [ ] Training time reasonable

### Evidence Saved
- [ ] Training log
- [ ] Loss/accuracy plot
- [ ] Final metrics (JSON/CSV)
- [ ] Inference test output
```

---

## Acceptance Criteria

### Story is DONE When:

1. ✅ **All Acceptance Criteria Met**
   - Every checkbox in story's acceptance criteria checked
   - No exceptions or "mostly done"

2. ✅ **Definition of Done Complete**
   - Every item in DOD checklist verified
   - Evidence exists for each item

3. ✅ **Validation Proof Documented**
   - Appropriate validation type performed
   - Evidence saved in `validation/` folder
   - File naming convention followed

4. ✅ **PROGRESS_TRACKER Updated**
   - Story marked as complete
   - Validation proof noted
   - Metrics updated

5. ✅ **Git Commit Made**
   - Commit message format: `[TVLJ-###] ✅ Story title`
   - All changes committed
   - No uncommitted work

6. ✅ **Can Demonstrate**
   - Can show it working
   - Can explain how it works
   - Someone else could reproduce it

---

## Quality Standards

### Minimum Standards for DONE

**Code**:
- Runs without errors
- Tests pass
- Coverage ≥80%
- No linting errors
- Type hints present

**Documentation**:
- Meets word count
- All sections complete
- Renders correctly
- Technically accurate

**Training**:
- Meets accuracy target
- Loss converges
- Can reproduce results

**Deployment**:
- Service accessible
- Responds correctly
- Error handling works
- Monitored

---

## Example: Complete Validation

### Story 2.1: Implement Transformer

**Validation Performed**:
```bash
# 1. Run implementation
python src/models/transformer.py > validation/phase2/story-2.1-output.txt
# Output: Model forward pass successful

# 2. Run tests
pytest tests/test_transformer.py -v --cov=src/models/transformer.py > validation/phase2/story-2.1-tests.txt
# Output: 15 tests passed, 87% coverage

# 3. Generate coverage report
pytest --cov=src/models/transformer.py --cov-report=html
# Screenshot saved: validation/phase2/story-2.1-coverage.png

# 4. Lint check
flake8 src/models/transformer.py > validation/phase2/story-2.1-lint.txt
# Output: No errors

# 5. Type check
mypy src/models/transformer.py > validation/phase2/story-2.1-mypy.txt
# Output: Success: no issues found

# 6. Update tracker
# Updated PROGRESS_TRACKER.md: Story 2.1 ✅ COMPLETE

# 7. Git commit
git add .
git commit -m "[TVLJ-201] ✅ Implement complete Transformer model"
git push
```

**Evidence Files**:
```
validation/phase2/
├── story-2.1-output.txt          # ✅ Runs successfully
├── story-2.1-tests.txt           # ✅ 15/15 tests pass
├── story-2.1-coverage.png        # ✅ 87% coverage
├── story-2.1-lint.txt            # ✅ No errors
└── story-2.1-mypy.txt            # ✅ Type check passed
```

**Result**: Story 2.1 is DONE ✅

---

## Best Practices

1. **Validate as you go**: Don't wait until the end
2. **Save evidence immediately**: Don't rely on memory
3. **Be thorough**: Better to over-document than under
4. **Use automation**: Scripts for common validations
5. **Screenshot everything**: Visual proof is powerful
6. **Keep organized**: Follow naming conventions
7. **Update tracker promptly**: Real-time progress tracking

---

## Automation Scripts

### Validate Code Story
```bash
#!/bin/bash
# validate_code_story.sh STORY_ID FILE_PATH

STORY_ID=$1
FILE_PATH=$2
PHASE=$(echo $STORY_ID | cut -d'.' -f1 | sed 's/^0*//')
VALIDATION_DIR="validation/phase$PHASE"

mkdir -p $VALIDATION_DIR

echo "🔍 Validating $STORY_ID: $FILE_PATH"

# Run file
python $FILE_PATH > $VALIDATION_DIR/story-$STORY_ID-output.txt 2>&1

# Run tests
pytest tests/test_$(basename $FILE_PATH) -v --cov=$FILE_PATH \
  > $VALIDATION_DIR/story-$STORY_ID-tests.txt 2>&1

# Lint
flake8 $FILE_PATH > $VALIDATION_DIR/story-$STORY_ID-lint.txt 2>&1

# Type check
mypy $FILE_PATH > $VALIDATION_DIR/story-$STORY_ID-mypy.txt 2>&1

echo "✅ Validation complete. Check $VALIDATION_DIR/"
```

---

**Remember: "If it's not validated, it's not done."**

Use this guide for every story to ensure quality and completeness.

Last Updated: 2026-01-31
