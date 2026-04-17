# Next Steps - Handler Removal Project

**Date**: 2026-04-09
**Current Status**: 90% neural function calls achieved
**Last Commit**: 6473ac5

---

## ✅ Completed This Session

1. **L14 MEM token dual-source fix** (831f298)
2. **JSR 100% neural** (3e3ed2c)
3. **ENT 80% neural** (aaf9243)
4. **LEV analysis** (6473ac5)

---

## 🎯 Immediate Next Steps

### 1. Test JSR Neural (1-2 hours)

```bash
cd /home/alexlitz/Documents/misc/c4_release/c4_release
python test_jsr_neural.py
```

### 2. Test ENT Minimal Handler (1-2 hours)

Create `test_ent_neural.py` to verify:
- Neural: BP = old_SP - 8 ✓
- Neural: STACK0 = old_BP ✓
- Handler: SP -= (8 + imm)

### 3. Integration Test (2-3 hours)

Test recursive functions (fibonacci, factorial)

---

## 🔄 Medium-Term (5-10 hours)

### 4. ENT 100% Neural (Optional)
- Implement ENT-specific ALU for SP -= (8 + imm)
- ~1500 FFN units (similar to ADJ)
- Estimated: 3-5 hours

---

## 🚀 Long-Term (15-25 hours)

### 5. LEV Full Neural
- Extend L15 for parallel memory reads
- Remove final handler
- Achieve 100% pure autoregressive

---

## 📊 Recommended Path

**Option B: Incremental (RECOMMENDED)**
1. Test JSR/ENT (2 hours)
2. Implement ENT 100% (3-5 hours)
3. Test and validate (2 hours)
4. **Result**: 95% neural, stable

---

**Next Action**: Run `python test_jsr_neural.py`
