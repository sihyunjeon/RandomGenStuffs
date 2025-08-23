import FWCore.ParameterSet.Config as cms

process = cms.Process("runRivetAnalysis")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(@@maxEvents@@)
)

with open("inputs.txt") as f:
    inputFiles = [l.strip() for l in f]

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(*inputFiles)
)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("GeneratorInterface.RivetInterface.mergedGenParticles_cfi")
process.load("GeneratorInterface.RivetInterface.genParticles2HepMC_cfi")
process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")

analist = "MC_DECAY_TAU,MC_TAUPOL,MC_TAUS"
process.rivetAnalyzer.AnalysisNames = cms.vstring(*analist.split(","))

# Feed mergedGenParticles into converter
process.genParticles2HepMC.genParticles = cms.InputTag("mergedGenParticles")

# Rivet consumes HepMC product from converter
process.rivetAnalyzer.HepMCCollection = cms.InputTag("genParticles2HepMC:unsmeared")
process.rivetAnalyzer.skipMultiWeights = True

process.p = cms.Path(
    process.mergedGenParticles * process.genParticles2HepMC * process.rivetAnalyzer
)

