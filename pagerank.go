// Copyright 2025 The aedoom Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"sync/atomic"
)

const (
	// Cycles is the number of frames per cycle
	Cycles = 30
	// Pixels is the number of pixel values
	Pixels = 256
	// Acts is index where the array of action ranks end
	Acts = Pixels + int(ActionCount)
	// Ranks is the size of the ranks array
	Ranks = Acts + Cycles
)

// PageRank is a pagerank based model
type PageRank struct {
	Rng       *rand.Rand
	Markov    [][Ranks][Ranks]uint64
	W, H      int
	Action    TypeAction
	Votes     [ActionCount]uint64
	Started   bool
	Iteration int
	Loop      []chan bool
}

// NewPageRank new pagerank model
func NewPageRank() *PageRank {
	p := &PageRank{
		Rng: rand.New(rand.NewSource(1)),
	}
	return p
}

// Process processes a frame
func (p *PageRank) Process(img Frame) (bool, TypeAction) {
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if p.Markov == nil {
		w, h := width/8, height/8
		fmt.Println(width, height, w, h, w*h)
		p.Markov = make([][Ranks][Ranks]uint64, w*h)
		p.Loop = make([]chan bool, w*h)
		for i := range p.Loop {
			p.Loop[i] = make(chan bool, 8)
		}
		p.W, p.H = w, h
	}
	index, previous := 0, img.GrayAt(0, 0).Y
	for y := 0; y < height-8; y += 8 {
		for x := 0; x < width-8; x += 8 {
			for yy := 0; yy < 8; yy++ {
				for xx := 0; xx < 8; xx++ {
					current := img.GrayAt(x+xx, y+yy).Y
					p.Markov[index][previous][current]++
					p.Markov[index][256+p.Action][current]++
					for i := range ActionCount {
						p.Markov[index][current][Pixels+i]++
					}
					p.Markov[index][current][Pixels+p.Action]++
					p.Markov[index][current][Acts+p.Iteration%Cycles]++
					p.Markov[index][Acts+p.Iteration][current%Cycles]++
					previous = current
				}
			}
			index++
		}
	}

	process := func(loop chan bool, seed int64, index int) {
		rng := rand.New(rand.NewSource(seed))
		current := 0
		var ranks [Ranks]uint64
		for range loop {
			for range 8 * 256 {
				sum := uint64(0)
				for _, value := range p.Markov[index][current] {
					sum += value
				}
				if sum == 0 {
					current = rng.Intn(Ranks)
					continue
				}
				total, selected := uint64(0), uint64(rng.Intn(int(sum)))
				for i, value := range p.Markov[index][current] {
					total += value
					if selected < total {
						ranks[i]++
						current = i
						break
					}
				}
			}
			r := ranks[Pixels:Acts]
			sum := uint64(0)
			for _, value := range r {
				sum += value
			}
			if sum == 0 {
				atomic.AddUint64(&p.Votes[rng.Intn(len(p.Votes))], 1)
				continue
			}
			total, selected := uint64(0), uint64(rng.Intn(int(sum)))
			for i, value := range r {
				total += value
				if selected < total {
					atomic.AddUint64(&p.Votes[i], 1)
					break
				}
			}
			shift := false
			for i := range ranks {
				if ranks[i] > 256*256 {
					shift = true
					break
				}
			}
			if shift {
				for i := range ranks {
					ranks[i] >>= 1
				}
			}
		}
	}
	if !p.Started {
		for i := range p.W * p.H {
			go process(p.Loop[i], p.Rng.Int63(), i)
		}
		p.Started = true
		for i := range p.Loop {
			p.Loop[i] <- true
		}
	}

	p.Iteration++
	sum := uint64(0)
	for _, value := range p.Votes {
		sum += value
	}
	if p.Iteration >= Cycles && sum >= uint64(p.W*p.H) {
		total, selected := uint64(0), uint64(p.Rng.Intn(int(sum)))
		for i, value := range p.Votes {
			total += value
			if selected < total {
				p.Action = TypeAction(i)
				break
			}
		}
		for i := range p.Votes {
			atomic.StoreUint64(&p.Votes[i], 0)
		}
		shift := false
	outer:
		for i := range p.Markov {
			for ii := range p.Markov[i] {
				for iii := range p.Markov[i][ii] {
					if p.Markov[i][ii][iii] > 256*256 {
						shift = true
						break outer
					}
				}
			}
		}
		if shift {
			for i := range p.Markov {
				for ii := range p.Markov[i] {
					for iii := range p.Markov[i][ii] {
						p.Markov[i][ii][iii] >>= 1
					}
				}
			}
		}
		for i := range p.Loop {
			p.Loop[i] <- true
		}
		p.Iteration = 0
		return true, p.Action
	}
	return false, ActionCount
}
