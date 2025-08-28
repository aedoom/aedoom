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

// Morpheus is the morpheus model
type Morpheus struct {
	rng       *rand.Rand
	iteration int
	w, h      int
	context   int
	buffers   [][16][16]byte
	state     []Matrix[float64]
	votes     [16]int
}

// NewMorpheus creates a new morpheus model
func NewMorpheus() *Morpheus {
	morpheus := Morpheus{}
	morpheus.rng = rand.New(rand.NewSource(1))
	return &morpheus
}

// Process processes a frame
func (m *Morpheus) Process(img Frame) (bool, TypeAction) {
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if m.buffers == nil {
		w, h := width/4, height/4
		fmt.Println(width, height, w, h, w*h)
		m.buffers = make([][16][16]byte, w*h)
		for i := range m.buffers {
			for ii := range m.buffers[i] {
				for iii := range m.buffers[i][ii] {
					m.buffers[i][ii][iii] = byte(m.rng.Intn(256))
				}
			}
		}
		m.state = make([]Matrix[float64], w*h)
		for i := range m.state {
			m.state[i] = NewMatrix(16, 16, make([]float64, 16*16)...)
			for ii := range m.state[i].Data {
				m.state[i].Data[ii] = m.rng.Float64()
			}
		}
		m.w, m.h = w, h
	}
	index := 0
	for y := 0; y < height-4; y += 4 {
		for x := 0; x < width-4; x += 4 {
			for yy := 0; yy < 4; yy++ {
				for xx := 0; xx < 4; xx++ {
					m.buffers[index][m.context][yy*4+xx] = img.GrayAt(x+xx, y+yy).Y
				}
			}
			index++
		}
	}

	indexes := rand.Perm((m.w - 1) * (m.h - 1))
	indexes = indexes[:len(indexes)/(4*Scale)]

	type Vote struct {
		V int
	}
	done := make(chan Vote, 8)
	process := func(i int, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		var v Vote
		a := m.state[i]
		b := NewMatrix(16, 16, make([]float64, 16*16)...)
		idx := 0
		for ii := range m.buffers[i] {
			for iii := range m.buffers[i][ii] {
				b.Data[idx] = float64(m.buffers[i][ii][iii]) / 255.0
				idx++
			}
		}

		const iterations = 8
		results := make([][]float64, iterations)
		for iteration := range iterations {
			x, y := NewMatrix(16, 16, make([]float64, 16*16)...), NewMatrix(16, 16, make([]float64, 16*16)...)
			index := 0
			for range x.Rows {
				for range x.Cols {
					x.Data[index] = rng.NormFloat64()
					y.Data[index] = rng.NormFloat64()
					index++
				}
			}
			x = x.Softmax(1)
			y = y.Softmax(1)
			a := x.MulT(a)
			b := y.MulT(b)
			graph := pagerank.NewGraph()
			for ii := range a.Rows {
				x := NewMatrix(16, 1, a.Data[ii*a.Cols:(ii+1)*a.Cols]...)
				for iii := range b.Rows {
					y := NewMatrix(16, 1, b.Data[iii*b.Cols:(iii+1)*b.Cols]...)
					cs := x.CS(y)
					if cs < 0 {
						cs = -cs
					}
					if math.IsNaN(cs) {
						panic(cs)
					}
					graph.Link(uint32(ii), uint32(iii), cs)
				}
			}
			result := make([]float64, 16)
			graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
				result[node] = rank
			})
			results[iteration] = result
		}
		avg := make([]float64, 16)
		for _, result := range results {
			for i, value := range result {
				avg[i] += value
			}
		}
		for i, value := range avg {
			avg[i] = value / float64(iterations)
		}

		stddev := make([]float64, 16)
		for _, result := range results {
			for i, value := range result {
				diff := value - avg[i]
				stddev[i] += diff * diff
			}
		}
		for i, value := range stddev {
			stddev[i] = math.Sqrt(value / float64(iterations))
		}

		min := float64(math.MaxFloat64)
		for i, value := range stddev {
			if value < min {
				min, v.V = value, i
			}
		}

		cov := make([][]float64, 16)
		for i := range cov {
			cov[i] = make([]float64, 16)
		}
		for _, measures := range results {
			for i, v := range measures {
				for ii, vv := range measures {
					diff1 := avg[i] - v
					diff2 := avg[ii] - vv
					cov[i][ii] += diff1 * diff2
				}
			}
		}
		if len(results) > 0 {
			for i := range cov {
				for ii := range cov[i] {
					cov[i][ii] = cov[i][ii] / float64(len(results))
				}
			}
		}
		index := 0
		for ii := range cov {
			for iii := range cov[ii] {
				m.state[i].Data[index] = cov[ii][iii]
				index++
			}
		}
		done <- v
	}
	index, flight, cpus := 0, 0, runtime.NumCPU()
	for index < len(indexes) && flight < cpus {
		go process(indexes[index], m.rng.Int63())
		flight++
		index++
	}
	for index < len(indexes) {
		v := <-done
		if v.V >= 0 {
			m.votes[v.V] += 1
		}
		flight--

		go process(indexes[index], m.rng.Int63())
		flight++
		index++
	}
	for range flight {
		v := <-done
		if v.V >= 0 {
			m.votes[v.V] += 1
		}
	}
	m.context = (m.context + 1) % 16
	m.iteration++
	if m.iteration%30 == 0 {
		max, index := 0, TypeAction(0)
		for i := range ActionCount {
			if m.votes[i+1] > max {
				max, index = m.votes[i+1], i
			}
		}
		for i := range m.votes {
			m.votes[i] = 0
		}
		return true, index
	}
	return false, ActionCount
}
