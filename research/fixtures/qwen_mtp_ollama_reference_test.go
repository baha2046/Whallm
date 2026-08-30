package qwen4_exp

import (
	"os"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

func TestMTPExternalReference(t *testing.T) {
	input := os.Getenv("QWEN_MTP_REFERENCE_INPUT")
	output := os.Getenv("QWEN_MTP_REFERENCE_OUTPUT")
	configPath := os.Getenv("QWEN_MTP_REFERENCE_CONFIG")
	if input == "" || output == "" || configPath == "" {
		t.Skip("external reference paths are not set")
	}

	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := parseConfig(configData)
	if err != nil {
		t.Fatal(err)
	}

	tensors := make(map[string]*mlx.Array)
	for name, value := range mlx.Load(input) {
		tensors[name] = value
	}
	if tensors["reference.embedding.weight"] == nil || tensors["reference.target_hidden"] == nil {
		t.Fatal("reference input tensors are missing")
	}

	m := &Model{
		Config:      &cfg,
		Layers:      make([]*Layer, cfg.NumHiddenLayers),
		quantBits:   4,
		quantGroup:  32,
		quantMode:   "mxfp4",
		tensorQuant: nil,
	}
	linears := model.NewLinearFactory(tensors, 32, 4, "mxfp4", nil)
	m.MTP, err = m.loadMTP(linears, tensors)
	if err != nil {
		t.Fatal(err)
	}
	m.EmbedTokens = nn.NewEmbedding(tensors["reference.embedding.weight"])

	b := &batch.Batch{
		InputIDs:     mlx.FromValues([]int32{0}, 1, 1),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
		Hidden:       tensors["reference.target_hidden"],
	}
	embedding := m.EmbedTokens.Forward(b.InputIDs)
	projectedEmbedding := m.MTP.FCEmbedding.Forward(m.MTP.EmbeddingNorm.Forward(embedding, m.RMSNormEps))
	projectedHidden := m.MTP.HiddenNorm.Forward(b.Hidden, m.RMSNormEps)
	projectedHidden = mlx.Reshape(projectedHidden, 1, 1, m.HCCount, m.HiddenSize)
	projectedHidden = m.MTP.FCHidden.Forward(projectedHidden)
	layerInput := mlx.Reshape(mlx.Add(projectedHidden, mlx.ExpandDims(projectedEmbedding, -2)), 1, 1, m.HCCount*m.HiddenSize)

	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	ropePositions := canonicalRopePositionRows(b, 1, nil)
	attentionInput, attentionState := m.MTP.Layer.AttentionConnection.Prepare(layerInput, m.Config)
	attentionOutput := m.MTP.Layer.Attention.Forward(
		attentionInput,
		b,
		cache.NewKVCache(),
		cache.NewKVCache(),
		positions,
		ropePositions,
		m.Config,
	)
	afterAttention := m.MTP.Layer.AttentionConnection.Inject(attentionState, attentionOutput, m.Config)
	moeInput, moeState := m.MTP.Layer.MLPConnection.Prepare(afterAttention, m.Config)
	moeOutput := m.MTP.Layer.MoE.Forward(moeInput, m.Config)
	multi := m.MTP.Layer.MLPConnection.Inject(moeState, moeOutput, m.Config)
	sample := m.MTP.Mixer.Reduce(multi, m.Config)
	mlx.Eval(sample, multi, projectedEmbedding, projectedHidden, layerInput, attentionInput, attentionOutput, afterAttention, moeInput, moeOutput)
	if err := mlx.SaveSafetensors(output, map[string]*mlx.Array{
		"reference.sample_hidden":      sample,
		"reference.wide_hidden":        multi,
		"stage.01_projected_embedding": projectedEmbedding,
		"stage.02_projected_hidden":    projectedHidden,
		"stage.03_layer_input":         layerInput,
		"stage.04_attention_input":     attentionInput,
		"stage.05_attention_output":    attentionOutput,
		"stage.06_after_attention":     afterAttention,
		"stage.07_moe_input":           moeInput,
		"stage.08_moe_output":          moeOutput,
		"stage.09_wide_hidden":         multi,
	}); err != nil {
		t.Fatal(err)
	}
}
