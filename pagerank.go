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
	Iteration int
	Loop      []chan Frame
}

// NewPageRank new pagerank model
func NewPageRank() *PageRank {
	p := &PageRank{
		Rng: rand.New(rand.NewSource(1)),
	}
	return p
}

func (p *PageRank) process(loop chan Frame, seed int64, index int, x, y int) {
	rng := rand.New(rand.NewSource(seed))
	current := 0
	var ranks [Ranks]uint64
	for {
		for img := range loop {
			previous := img.GrayAt(x, y).Y
			for yy := 0; yy < 8; yy++ {
				for xx := 0; xx < 8; xx++ {
					current := byte(0)
					if y&1 == 0 {
						current = img.GrayAt(x+xx, y+yy).Y
					} else {
						current = img.GrayAt(7-x+xx, y+yy).Y
					}
					p.Markov[index][previous][current]++
					p.Markov[index][256+p.Action][current]++
					for i := range ActionCount {
						p.Markov[index][current][Pixels+i]++
					}
					p.Markov[index][current][Pixels+p.Action]++
					p.Markov[index][current][Acts+p.Iteration%Cycles]++
					p.Markov[index][Acts+p.Iteration%Cycles][current]++
					previous = current
				}
			}
			for range 64 {
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
			if p.Iteration >= Cycles {
				break
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
		shift = false
	outer:
		for ii := range p.Markov[index] {
			for iii := range p.Markov[index][ii] {
				if p.Markov[index][ii][iii] > 256*256 {
					shift = true
					break outer
				}
			}
		}
		if shift {
			for ii := range p.Markov[index] {
				for iii := range p.Markov[index][ii] {
					p.Markov[index][ii][iii] >>= 1
				}
			}
		}
	}
}

// Process processes a frame
func (p *PageRank) Process(img Frame) (bool, TypeAction) {
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if p.Markov == nil {
		w, h := width/8, height/8
		fmt.Println(width, height, w, h, w*h)
		p.Markov = make([][Ranks][Ranks]uint64, w*h)
		p.Loop = make([]chan Frame, w*h)
		fmt.Println(w * h)
		for i := range p.Loop {
			p.Loop[i] = make(chan Frame, 8)
		}
		p.W, p.H = w, h

		index := 0
		for y := 0; y < height; y += 8 {
			for x := 0; x < width; x += 8 {
				go p.process(p.Loop[index], p.Rng.Int63(), index, x, y)
				index++
			}
		}
		fmt.Println(index)
	}

	for i := range p.Loop {
		p.Loop[i] <- img
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
		p.Iteration = 0
		return true, p.Action
	}
	return false, ActionCount
}
