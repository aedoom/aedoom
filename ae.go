// Copyright 2025 The aedoom Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/alixaxel/pagerank"
)

// AE is an autoencoder model
type AE struct {
	rng       *rand.Rand
	auto      [][Actions]Auto
	mind      [Actions]Auto
	votes     [Actions]float32
	counts    [Actions]float32
	iteration int
	state     State
	w, h      int
	input     []float32
	output    []float32
	index     int
	circular  [8]Matrix[float32]
}

// NewAE creates a new ae model
func NewAE() *AE {
	ae := AE{}
	ae.rng = rand.New(rand.NewSource(1))
	for i := range ae.mind {
		ae.mind[i].Auto = NewAutoEncoder(8, true)
		ae.mind[i].Action = TypeAction(i)
	}
	ae.input, ae.output = make([]float32, 8), make([]float32, 8)
	for i := range ae.circular {
		m := NewMatrix(Actions, 1, make([]float32, Actions)...)
		for ii := range m.Data {
			m.Data[ii] = ae.rng.Float32()
		}
		ae.circular[i] = m
	}
	return &ae
}

// Process processes a frame
func (a *AE) Process(img Frame) (bool, TypeAction) {
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if a.auto == nil {
		w, h := width/8, height/8
		fmt.Println(width, height, w, h, w*h)
		a.auto = make([][Actions]Auto, w*h)
		for i := range a.auto {
			for ii := range a.auto[i] {
				a.auto[i][ii].Auto = NewAutoEncoder(8*8, true)
				a.auto[i][ii].Action = TypeAction(ii)
			}
		}
		a.w, a.h = w, h
	}
	type Patch struct {
		Input   []float32
		Output  []float32
		Entropy float32
	}
	pixels := make([]Patch, 0, 8)
	for y := 0; y < height-8; y += 8 {
		for x := 0; x < width-8; x += 8 {
			input, output := make([]float32, 8*8), make([]float32, 8*8)
			for yy := 0; yy < 8; yy++ {
				for xx := 0; xx < 8; xx++ {
					pixel := float32(img.GrayAt(x+xx, y+yy).Y) / 255
					output[yy*8+xx] = pixel
					input[yy*8+xx] = pixel
				}
			}
			pixels = append(pixels, Patch{
				Input:  input,
				Output: output,
			})
		}
	}

	indexes := rand.Perm((a.w - 1) * (a.h - 1))
	indexes = indexes[:len(indexes)/Scale]

	type Vote struct {
		Min     int
		Max     int
		Entropy float32
	}
	done := make(chan Vote, 8)
	measure := func(i int, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		min, max, minIndex, maxIndex := float32(math.MaxFloat32), float32(0), 0, 0
		for ii := range a.auto[i] {
			value := a.auto[i][ii].Auto.Measure(pixels[i].Input, pixels[i].Output, &a.state)
			if value < min {
				min, minIndex = value, ii
			}
			if value > max {
				max, maxIndex = value, ii
			}
		}
		a.auto[i][maxIndex].Auto.Encode(pixels[i].Input, pixels[i].Output, rng, &a.state)
		done <- Vote{
			Min:     minIndex,
			Max:     maxIndex,
			Entropy: max,
		}
	}
	index, flight, cpus := 0, 0, runtime.NumCPU()
	for index < len(indexes) && flight < cpus {
		go measure(indexes[index], a.rng.Int63())
		flight++
		index++
	}
	for index < len(indexes) {
		act := <-done
		if act.Max >= 0 {
			a.votes[act.Max] += act.Entropy
			a.counts[act.Max]++
		}
		flight--

		go measure(indexes[index], a.rng.Int63())
		flight++
		index++
	}
	for range flight {
		act := <-done
		if act.Max >= 0 {
			a.votes[act.Max] += act.Entropy
			a.counts[act.Max]++
		}
	}
	a.iteration++
	if a.iteration%30 == 0 {
		for i := range a.votes {
			if a.counts[i] == 0 {
				continue
			}
			a.votes[i] /= a.counts[i]
		}
		copy(a.circular[a.index].Data, a.votes[:])
		a.index = (a.index + 1) % len(a.circular)
		graph := pagerank.NewGraph()
		for i := range a.circular {
			for ii := range a.circular {
				x, y := (i+a.index)%len(a.circular), (ii+a.index)%len(a.circular)
				cs := a.circular[x].CS(a.circular[y])
				graph.Link(uint32(x), uint32(y), float64(cs))
			}
		}
		graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			a.input[node] = float32(rank)
			a.output[node] = float32(rank)
		})
		max, action := float32(0.0), TypeAction(0)
		for i := range a.mind {
			value := a.mind[i].Auto.Measure(a.input, a.output, &a.state)
			if value > max {
				max, action = value, a.mind[i].Action
			}
		}
		a.mind[action].Auto.Encode(a.input, a.output, a.rng, &a.state)
		for i := range a.votes {
			a.votes[i] = 0
			a.counts[i] = 0
		}
		pre := TypeAction(action)
		for ii, value := range a.state {
			a.state[ii], pre = pre, value
		}
		return true, action
	}
	return false, ActionCount
}
