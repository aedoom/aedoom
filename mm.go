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

// MMWidth is the width of the morpheus model
const MMWidth = 16 + Actions

// MarkovOrder is the order of the markov model
const MarkovOrder = 4

// Markov is a markov state
type Markov [MarkovOrder]TypeAction

// Morpheus is the morpheus model
type MorpheusMarkov struct {
	rng       *rand.Rand
	iteration int
	w, h      int
	context   int
	buffers   [][MMWidth][MMWidth]byte
	actions   [][MMWidth][Actions]int
	state     Markov
	markov    map[Markov][Actions]int
}

// NewMorpheusMarkov creates a new morpheus markov model
func NewMorpheusMarkov() *MorpheusMarkov {
	fmt.Println("Morpheus Markov Mode")
	morpheus := MorpheusMarkov{}
	morpheus.rng = rand.New(rand.NewSource(1))
	morpheus.markov = make(map[Markov][Actions]int)
	return &morpheus
}

// Process processes a frame
func (m *MorpheusMarkov) Process(img Frame) (bool, TypeAction) {
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if m.buffers == nil {
		w, h := width/4, height/4
		fmt.Println(width, height, w, h, w*h)
		m.buffers = make([][MMWidth][MMWidth]byte, w*h)
		for i := range m.buffers {
			for ii := range m.buffers[i] {
				for iii := range m.buffers[i][ii] {
					m.buffers[i][ii][iii] = byte(m.rng.Intn(256))
				}
			}
		}
		m.actions = make([][MMWidth][Actions]int, w*h)
		for i := range m.actions {
			for ii := range m.actions[i] {
				for iii := range m.actions[i][ii] {
					m.actions[i][ii][iii] = m.rng.Intn(1024)
				}
			}
		}
		m.w, m.h = w, h
	}
	votes := m.markov[m.state]
	index := 0
	for y := 0; y < height-4; y += 4 {
		for x := 0; x < width-4; x += 4 {
			for yy := 0; yy < 4; yy++ {
				for xx := 0; xx < 4; xx++ {
					m.buffers[index][m.context][yy*4+xx] = img.GrayAt(x+xx, y+yy).Y
				}
			}
			for i := range Actions {
				m.buffers[index][m.context][3*4+3+i] = 0
			}
			sum := 0.0
			for i := range votes {
				sum += float64(votes[i])
			}
			for i := range votes {
				m.buffers[index][m.context][3*4+3+i] = byte(255 * float64(votes[i]) / sum)
				m.actions[index][m.context][i] = votes[i]
			}
			index++
		}
	}

	indexes := rand.Perm((m.w - 1) * (m.h - 1))
	indexes = indexes[:len(indexes)/(4*Scale)]

	done := make(chan [Actions]int, 8)
	process := func(i int, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		a := NewMatrix(MMWidth, MMWidth, make([]float32, MMWidth*MMWidth)...)
		idx := 0
		for ii := range m.buffers[i] {
			for iii := range m.buffers[i][ii] {
				a.Data[idx] = float32(m.buffers[i][ii][iii]) / 255.0
				idx++
			}
		}

		const iterations = 8
		results := make([][]float64, iterations)
		for iteration := range iterations {
			x, y := NewMatrix(MMWidth, MMWidth/2, make([]float32, MMWidth*MMWidth/2)...), NewMatrix(MMWidth, MMWidth/2, make([]float32, MMWidth*MMWidth/2)...)
			index := 0
			for range x.Rows {
				for range x.Cols {
					x.Data[index] = float32(rng.NormFloat64())
					y.Data[index] = float32(rng.NormFloat64())
					index++
				}
			}
			x = x.Softmax(1)
			y = y.Softmax(1)
			aa := x.MulT(a)
			bb := y.MulT(a)
			graph := pagerank.NewGraph()
			for ii := range aa.Rows {
				x := NewMatrix(MMWidth/2, 1, aa.Data[ii*aa.Cols:(ii+1)*aa.Cols]...)
				for iii := range bb.Rows {
					y := NewMatrix(MMWidth/2, 1, bb.Data[iii*bb.Cols:(iii+1)*bb.Cols]...)
					cs := x.CS(y)
					if cs < 0 {
						cs = -cs
					}
					if math.IsNaN(float64(cs)) {
						panic(cs)
					}
					graph.Link(uint32(ii), uint32(iii), float64(cs))
				}
			}
			result := make([]float64, MMWidth)
			graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
				result[node] = rank
			})
			results[iteration] = result
		}
		avg := make([]float64, MMWidth)
		for _, result := range results {
			for i, value := range result {
				avg[i] += value
			}
		}
		for i, value := range avg {
			avg[i] = value / float64(iterations)
		}

		stddev := make([]float64, MMWidth)
		for _, result := range results {
			for i, value := range result {
				diff := value - avg[i]
				stddev[i] += diff * diff
			}
		}
		for i, value := range stddev {
			stddev[i] = math.Sqrt(value / float64(iterations))
		}

		cov := make([][]float64, MMWidth)
		for i := range cov {
			cov[i] = make([]float64, MMWidth)
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

		vv := make([]float64, len(cov))
		for i := range cov {
			for _, value := range cov[i] {
				vv[i] = value * value
			}
		}
		for i, value := range vv {
			vv[i] = math.Sqrt(value)
		}

		min, vote := float64(math.MaxFloat64), 0
		for i, value := range vv {
			if value < min {
				min, vote = value, i
			}
		}
		done <- m.actions[i][vote]
	}
	index, flight, cpus := 0, 0, runtime.NumCPU()
	for index < len(indexes) && flight < cpus {
		go process(indexes[index], m.rng.Int63())
		flight++
		index++
	}
	for index < len(indexes) {
		v := <-done
		for i, value := range v {
			votes[i] += value
		}
		flight--

		go process(indexes[index], m.rng.Int63())
		flight++
		index++
	}
	for range flight {
		v := <-done
		for i, value := range v {
			votes[i] += value
		}
	}
	m.context = (m.context + 1) % MMWidth
	m.iteration++
	if m.iteration%30 == 0 {
		sum, act := 0.0, TypeAction(0)
		for i := range votes {
			sum += float64(votes[i])
		}
		total, selected := 0.0, m.rng.Float64()
		for i := range votes {
			total += float64(votes[i]) / sum
			if selected < total {
				act = TypeAction(i)
				break
			}
		}
		for {
			max := 0
			for i := range votes {
				if votes[i] > max {
					max = votes[i]
				}
			}
			if max < 8*1024 {
				break
			}
			for i := range votes {
				votes[i] >>= 1
				if votes[i] == 0 {
					votes[i] = 1
				}
			}
		}
		m.markov[m.state] = votes
		for i := range m.state {
			m.state[i], act = act, m.state[i]
		}
		return true, act
	}
	m.markov[m.state] = votes
	return false, ActionCount
}
